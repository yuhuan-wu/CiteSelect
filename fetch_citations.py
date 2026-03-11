#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import logging
import sys
import threading
from typing import Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from award_matcher import get_awards_matcher
from utils import (
    active_publications_jsonl,
    citations_paths_for_paper,
    legacy_citations_paths,
    RateLimiter,
    build_semanticscholar,
    load_state,
    save_state,
    setup_dirs,
    tsv_safe,
    author_names,
)


def fetch_author_metrics_batch(cfg: Dict[str, Any], author_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    调用 Semantic Scholar Graph API 批量获取作者指标（name, citationCount）。
    返回 {authorId: {"name": str, "citationCount": int}}
    """
    ids = [str(x) for x in author_ids if x]
    if not ids:
        return {}
    # 去重保持顺序
    ids = list(dict.fromkeys(ids))

    url = "https://api.semanticscholar.org/graph/v1/author/batch"
    params = {"fields": "authorId,name,citationCount"}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    api_key = cfg.get("s2_api_key")
    if api_key:
        headers["x-api-key"] = api_key

    out: Dict[str, Dict[str, Any]] = {}
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        resp = requests.post(url, params=params, json={"ids": chunk}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 兼容返回为 list 或 {"data": [...]}
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for it in items:
                try:
                    aid = it.get("authorId") or it.get("id") or (it.get("author") or {}).get("authorId")
                    if aid:
                        out[str(aid)] = {"name": it.get("name"), "citationCount": it.get("citationCount")}
                except Exception:
                    continue
    return out


def fetch_author_metrics_single(cfg: Dict[str, Any], author_id: str) -> Optional[Dict[str, Any]]:
    """
    单个作者回退查询：当批量接口缺失或返回 0 时，尝试单查修正。
    返回 {"name": str, "citationCount": int} 或 None
    """
    if not author_id:
        return None
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    params = {"fields": "name,citationCount"}
    headers = {"accept": "application/json"}
    api_key = cfg.get("s2_api_key")
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return {"name": data.get("name"), "citationCount": data.get("citationCount")}
    except Exception:
        return None
    return None


def fetch_citations_for_paper(
    cfg: Dict[str, Any],
    paper_id: str,
    force: bool = False,
    per_paper_max: Optional[int] = None,
    rate_limiter: Any = None,
    retries: int = 3,
) -> bool:
    """
    抓取单篇论文的引用列表，写入 citations/{paperId}.jsonl 与 .txt
    逻辑与原 collect.py 中实现保持一致
    """
    jsonl_path_new, txt_path_new = citations_paths_for_paper(cfg, paper_id)
    jsonl_path_old, txt_path_old = legacy_citations_paths(paper_id)
    if not force and (
        (jsonl_path_new.exists() and txt_path_new.exists())
        or (jsonl_path_old.exists() and txt_path_old.exists())
    ):
        logging.info(f"跳过 {paper_id}，已存在 citations 文件")
        return True

    s2 = build_semanticscholar(cfg)
    limit = int(cfg.get("page_size", 100))
    attempt = 0
    backoff = 1.0
    # 缓存作者指标，避免重复请求
    author_metrics_cache: Dict[str, Dict[str, Any]] = {}
    awards_cfg = cfg.get("awards") or {}
    awards_enabled = bool(awards_cfg.get("enabled", True))
    awards_max_workers = int(awards_cfg.get("max_workers", 8))
    awards_matcher = get_awards_matcher(cfg) if awards_enabled else None

    while attempt <= max(0, int(retries)):
        try:
            if rate_limiter:
                rate_limiter.acquire()
            logging.info(f"抓取论文 {paper_id} 的引用列表...")
            # 修复 400：语义学者 Python SDK 会对传入的 fields 做 ','.join(fields)，
            # 若传入 str 则被当作字符序列导致 c,o,n,t,... 这种非法字段。
            # 这里恢复为 None 使用 SDK 的默认字段集合（此前验证可 200）。
            results = s2.get_paper_citations(
                paper_id=paper_id,
                fields=None,
                limit=limit,
            )
            count = 0
            with jsonl_path_new.open("w", encoding="utf-8") as fjson, txt_path_new.open("w", encoding="utf-8") as ftxt:
                for c in results:
                    paper = getattr(c, "paper", None)
                    if paper is None:
                        continue
                    edge = {
                        "isInfluential": getattr(c, "isInfluential", None),
                        "intents": getattr(c, "intents", None),
                        "contexts": getattr(c, "contexts", None),
                        "contextsWithIntent": getattr(c, "contextsWithIntent", None),
                    }

                    # 收集作者ID并获取作者被引指标（批量）
                    authors_objs: List[Any] = []
                    try:
                        authors_objs = list(getattr(paper, "authors", []) or [])
                    except Exception:
                        authors_objs = []

                    author_ids: List[str] = []
                    for a in authors_objs:
                        aid = getattr(a, "authorId", None)
                        # 尝试从 raw_data 中取
                        if not aid:
                            try:
                                raw_a = getattr(a, "raw_data", None)
                                if isinstance(raw_a, dict):
                                    aid = raw_a.get("authorId") or raw_a.get("id")
                                    if not aid:
                                        url = raw_a.get("url")
                                        if isinstance(url, str):
                                            m = re.search(r"/author/[^/]+/([^/?#]+)", url)
                                            if m:
                                                aid = m.group(1)
                            except Exception:
                                pass
                        # 再从对象的 url 属性尝试
                        if not aid:
                            try:
                                url2 = getattr(a, "url", None)
                                if isinstance(url2, str):
                                    m = re.search(r"/author/[^/]+/([^/?#]+)", url2)
                                    if m:
                                        aid = m.group(1)
                            except Exception:
                                pass
                        if isinstance(aid, (str, int)):
                            author_ids.append(str(aid))

                    authors_formatted: List[str] = []
                    author_signals: List[Dict[str, Any]] = []
                    if cfg.get("include_author_citations", True) and author_ids:
                        # 先批量查缺失
                        missing = [aid for aid in author_ids if aid not in author_metrics_cache]
                        if missing:
                            if rate_limiter:
                                rate_limiter.acquire()
                            try:
                                batch_map = fetch_author_metrics_batch(cfg, missing)
                                author_metrics_cache.update(batch_map)
                            except Exception:
                                # 忽略作者指标获取失败，降级为仅输出作者名
                                pass
                        # 对于缺失或为 0 的条目进行单查回退（可配置）
                        if cfg.get("author_metrics_fallback_single", True):
                            retry_zero = bool(cfg.get("author_metrics_retry_zero", True))
                            needs_fix: List[str] = []
                            for aid in author_ids:
                                rec = author_metrics_cache.get(aid)
                                if rec is None or (retry_zero and (rec.get("citationCount") in (None, 0))):
                                    needs_fix.append(aid)
                            for aid in needs_fix:
                                if rate_limiter:
                                    rate_limiter.acquire()
                                rec = fetch_author_metrics_single(cfg, aid)
                                if isinstance(rec, dict) and rec.get("citationCount") is not None:
                                    author_metrics_cache[aid] = rec

                    for a in authors_objs:
                        name = getattr(a, "name", None) or ""
                        aid = getattr(a, "authorId", None)
                        if not aid and hasattr(a, "raw_data"):
                            try:
                                raw_a = a.raw_data
                                if isinstance(raw_a, dict):
                                    aid = raw_a.get("authorId") or raw_a.get("id")
                                    if not aid:
                                        url = raw_a.get("url")
                                        if isinstance(url, str):
                                            m = re.search(r"/author/[^/]+/([^/?#]+)", url)
                                            if m:
                                                aid = m.group(1)
                            except Exception:
                                pass
                        if not aid:
                            try:
                                url2 = getattr(a, "url", None)
                                if isinstance(url2, str):
                                    m = re.search(r"/author/[^/]+/([^/?#]+)", url2)
                                    if m:
                                        aid = m.group(1)
                            except Exception:
                                pass
                        cnt = None
                        if aid is not None:
                            cnt = (author_metrics_cache.get(str(aid)) or {}).get("citationCount")
                        if name:
                            authors_formatted.append(f"{name} ({cnt})" if cnt is not None else name)
                        author_signals.append({
                            "name": name,
                            "authorId": str(aid) if aid is not None else None,
                            "citationCount": cnt,
                        })

                    max_author_citation = 0
                    for signal in author_signals:
                        cnt = signal.get("citationCount")
                        if isinstance(cnt, int):
                            max_author_citation = max(max_author_citation, cnt)

                    if awards_matcher and author_signals:
                        awards_map = awards_matcher.match_names(
                            [str(sig.get("name") or "") for sig in author_signals if sig.get("name")],
                            max_workers=awards_max_workers,
                        )
                        for signal in author_signals:
                            signal["awardMatches"] = awards_map.get(str(signal.get("name") or ""), [])

                    cite = {
                        "citingPaper": {
                            "paperId": paper.paperId,
                            "title": paper.title,
                            "year": paper.year,
                            "venue": paper.venue,
                            "citationCount": paper.citationCount,
                            "url": paper.url,
                            "externalIds": getattr(paper, "externalIds", None),
                            "authors": author_names(paper),
                        },
                        "edge": edge,
                        "authorSignals": author_signals,
                        "maxAuthorCitation": max_author_citation,
                        "raw": {"citingPaper": getattr(paper, "raw_data", None), "edge": edge},
                    }
                    fjson.write(json.dumps(cite, ensure_ascii=False) + "\n")

                    # 上下文摘要
                    contexts = edge.get("contexts") or []
                    ctx_snips: List[str] = []
                    for s in contexts[:3]:
                        s = s.strip() if isinstance(s, str) else ""
                        if len(s) > 300:
                            s = s[:300] + "…"
                        ctx_snips.append(s)

                    # TXT 行输出，作者为 “Name (count)” 形式
                    line = [
                        tsv_safe(edge.get("isInfluential")),
                        tsv_safe("|".join(edge.get("intents") or [])),
                        tsv_safe(paper.year),
                        tsv_safe(paper.citationCount),
                        tsv_safe(paper.title),
                        tsv_safe("; ".join(authors_formatted) if authors_formatted else "; ".join(author_names(paper))),
                        tsv_safe(paper.venue),
                        tsv_safe(paper.url),
                        tsv_safe(" || ".join(ctx_snips)),
                    ]
                    ftxt.write("\t".join(line) + "\n")
                    count += 1
                    if per_paper_max and count >= per_paper_max:
                        break
            logging.info(f"{paper_id} 引用数写入: {count}")
            return True
        except Exception as e:
            attempt += 1
            logging.warning(f"{paper_id} 抓取失败（第{attempt}次）：{e}")
            time.sleep(backoff)
            backoff *= 2

    return False


def fetch_citations(
    cfg: Dict[str, Any],
    only_first_n: Optional[int] = None,
    per_paper_max: Optional[int] = None,
) -> None:
    """
    批量抓取引用，支持并发与续跑，逻辑与原 collect.py 中实现保持一致
    """
    setup_dirs()
    force = bool(cfg.get("output", {}).get("force", False))
    pub_jsonl = active_publications_jsonl()
    if not pub_jsonl.exists():
        logging.error("缺少 publications.jsonl，请先运行 fetch-publications")
        sys.exit(1)  # type: ignore[name-defined]

    # 读取 paperIds
    paper_ids: List[str] = []
    with pub_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("paperId")
            if pid:
                paper_ids.append(pid)

    if only_first_n:
        paper_ids = paper_ids[:only_first_n]

    # 续跑过滤
    state = load_state()
    citations_done = state.get("citations_done", {})
    if cfg.get("output", {}).get("resume", True):
        filtered: List[str] = []
        for pid in paper_ids:
            new_jsonl, new_txt = citations_paths_for_paper(cfg, pid)
            old_jsonl, old_txt = legacy_citations_paths(pid)
            if citations_done.get(pid) and (
                (new_jsonl.exists() and new_txt.exists()) or (old_jsonl.exists() and old_txt.exists())
            ) and not force:
                logging.info(f"续跑跳过 {pid}（state 记录已完成）")
                continue
            filtered.append(pid)
        paper_ids = filtered

    if not paper_ids:
        logging.info("无待处理论文")
        return

    # 全局限流与并发
    rate_limiter = RateLimiter(cfg.get("s2_qps", 3))
    max_workers = int(cfg.get("max_concurrency", 3))
    state_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures: Dict[Any, str] = {}
        for pid in paper_ids:
            futures[ex.submit(
                fetch_citations_for_paper,
                cfg, pid, force, per_paper_max, rate_limiter, int(cfg.get("retry", 3))
            )] = pid

        completed = 0
        for fut in as_completed(futures):
            pid = futures[fut]
            ok = False
            try:
                ok = fut.result()
            except Exception as e:
                logging.exception(f"{pid} 任务异常: {e}")
            with state_lock:
                st = load_state()
                st.setdefault("citations_done", {})
                st["citations_done"][pid] = bool(ok)
                save_state(st)
            completed += 1
            logging.info(f"进度: {completed}/{len(futures)} 完成")
