#!/usr/bin/env python3
from __future__ import annotations

import re
import json
import time
import logging
import requests
from typing import Any, Dict, List, Optional

from utils import (
    active_publications_jsonl,
    LOGS_DIR,
    citations_paths_for_paper,
    citations_ref_paths_for_paper,
    legacy_citations_paths,
    tsv_safe,
    setup_dirs,
    RateLimiter,
)

from llm_client import create_llm_client_from_config
from refine_prompt import build_refine_prompt


# -----------------------------
# Author metrics helpers (for refine step)
# -----------------------------
def _fetch_author_metrics_batch(cfg: Dict[str, Any], author_ids: List[str], rate_limiter: Optional[RateLimiter] = None) -> Dict[str, Dict[str, Any]]:
    """
    批量查询作者指标（name, citationCount）。
    返回: {authorId: {"name": str, "citationCount": int}}
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
        if rate_limiter:
            rate_limiter.acquire()
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


def _fetch_author_metrics_single(cfg: Dict[str, Any], author_id: str, rate_limiter: Optional[RateLimiter] = None) -> Optional[Dict[str, Any]]:
    """
    单个作者查询（回退）：当批量结果缺失时补齐 name, citationCount
    """
    if not author_id:
        return None
    if rate_limiter:
        rate_limiter.acquire()
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


def _extract_author_pairs_from_raw(cp_raw: Any) -> List[Dict[str, Any]]:
    """
    从 raw.citingPaper 中提取作者 (authorId, name, url) 信息。
    返回: [{"authorId": str, "name": str, "url": str|None}, ...]
    """
    pairs: List[Dict[str, Any]] = []
    if not isinstance(cp_raw, dict):
        return pairs
    authors = (cp_raw.get("authors") or []) if isinstance(cp_raw.get("authors"), list) else []
    for a in authors:
        if not isinstance(a, dict):
            continue
        aid = a.get("authorId") or a.get("id")
        name = a.get("name")
        url = a.get("url")
        # 尝试从 url 解析 authorId
        if not aid and isinstance(url, str):
            import re as _re
            m = _re.search(r"/author/[^/]+/([^/?#]+)", url)
            if m:
                aid = m.group(1)
        if name:
            pairs.append({"authorId": str(aid) if aid is not None else None, "name": name, "url": url})
    return pairs


def _normalize_title_key(title: Any) -> str:
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", title.strip().lower())


def refine_citations_for_paper(cfg: Dict[str, Any], paper_id: str) -> None:
    """
    对单篇论文的 citations 进行 LLM 精排，输出到 citations_refined/ 中，
    命名规则与 citations/ 一致（默认使用“被引论文标题”清洗后的文件名；配置 output.file_naming 可切换）。
    """
    new_jsonl, _ = citations_paths_for_paper(cfg, paper_id)
    old_jsonl, _ = legacy_citations_paths(paper_id)
    active_jsonl = new_jsonl if new_jsonl.exists() else (old_jsonl if old_jsonl.exists() else None)
    if not active_jsonl:
        logging.warning(f"缺少 {new_jsonl} 与 {old_jsonl}，跳过")
        return

    model = cfg.get("llm", {}).get("model", "openai/gpt-5.4")
    temperature_cfg = cfg.get("llm", {}).get("temperature", 0.0)
    try:
        temperature: Optional[float] = float(temperature_cfg) if temperature_cfg is not None else None
    except Exception:
        temperature = None
    # gpt-5 系列不支持自定义 temperature，置为 None 以避免传参
    if isinstance(model, str):
        model_l = model.lower()
        if model_l.startswith("gpt-5") or model_l.startswith("openai/gpt-5"):
            temperature = None

    max_completion_tokens = int(cfg.get("llm", {}).get("max_tokens", 1024))
    batch_size = int(cfg.get("llm", {}).get("batch_size", 40))
    echo_stdout = bool(cfg.get("llm", {}).get("echo_stdout", True))

    # 初始化 OpenAI 客户端（模块化）
    client = create_llm_client_from_config(cfg)
    if not client.api_key:
        provider = str((cfg.get("llm") or {}).get("provider") or "openai").lower()
        env_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
        logging.error(f"{env_name} 未配置，无法执行精排")
        return

    # 读取目标论文元数据（从 publications.jsonl）
    target_meta: Dict[str, Any] = {}
    pub_jsonl = active_publications_jsonl()
    if pub_jsonl.exists():
        with pub_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("paperId") == paper_id:
                    target_meta = {"title": rec.get("title"), "year": rec.get("year"), "venue": rec.get("venue")}
                    break

    # 读取引用数据
    items: List[Dict[str, Any]] = []
    with active_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    # 计算本论文所有引用项的作者指标，并格式化为 "Name (count)"
    rate_limiter = RateLimiter(cfg.get("s2_qps", 3))
    author_metrics_cache: Dict[str, Dict[str, Any]] = {}
    include_author_citations = bool(cfg.get("include_author_citations", True))

    # 1) 收集每条引用的作者 (authorId, name)
    id_pairs_by_citing: Dict[str, List[Dict[str, Any]]] = {}
    all_author_ids: List[str] = []
    for it in items:
        cp = (it.get("citingPaper") or {})
        cp_raw = (it.get("raw") or {}).get("citingPaper")
        cid = cp.get("paperId")
        if not cid:
            continue
        pairs = _extract_author_pairs_from_raw(cp_raw)
        if not pairs:
            for sig in it.get("authorSignals") or []:
                if not isinstance(sig, dict):
                    continue
                pairs.append({
                    "authorId": sig.get("authorId"),
                    "name": sig.get("name"),
                    "url": None,
                })
        id_pairs_by_citing[cid] = pairs
        if include_author_citations:
            for p in pairs:
                aid = p.get("authorId")
                if aid:
                    all_author_ids.append(aid)
    # 去重
    all_author_ids = list(dict.fromkeys(all_author_ids))

    # 2) 批量获取作者指标
    if include_author_citations and all_author_ids:
        try:
            batch_map = _fetch_author_metrics_batch(cfg, all_author_ids, rate_limiter)
            author_metrics_cache.update(batch_map)
        except Exception:
            pass

    # 3) 对缺失的作者尝试单查回退（可按需扩展：cfg["author_metrics_fallback_single"] 控制）
    if include_author_citations and all_author_ids:
        try:
            fallback_enabled = bool(cfg.get("author_metrics_fallback_single", True))
        except Exception:
            fallback_enabled = True
        if fallback_enabled:
            for aid in all_author_ids:
                if aid in author_metrics_cache and (author_metrics_cache[aid].get("citationCount") is not None):
                    continue
                rec = _fetch_author_metrics_single(cfg, aid, rate_limiter)
                if isinstance(rec, dict) and (rec.get("citationCount") is not None):
                    author_metrics_cache[aid] = rec

    # 4) 基于缓存构造每条引用的格式化作者与最大被引
    authors_map: Dict[str, Dict[str, Any]] = {}  # {citingPaperId: {"authors_formatted": [str], "max_citation": int}}
    for it in items:
        cp = (it.get("citingPaper") or {})
        cid = cp.get("paperId")
        if not cid:
            continue
        pairs = id_pairs_by_citing.get(cid, []) or []
        formatted: List[str] = []
        max_cite = 0
        existing_signals = it.get("authorSignals") or []
        existing_count_by_name = {}
        for sig in existing_signals:
            if isinstance(sig, dict) and sig.get("name"):
                existing_count_by_name[str(sig.get("name"))] = sig.get("citationCount")
        for p in pairs:
            name = p.get("name") or ""
            aid = p.get("authorId")
            cnt = None
            if aid:
                cnt = (author_metrics_cache.get(str(aid)) or {}).get("citationCount")
            if cnt is None and name:
                cnt = existing_count_by_name.get(str(name))
            if isinstance(cnt, int) and cnt is not None:
                max_cite = max(max_cite, cnt)
            if name:
                formatted.append(f"{name} ({cnt})" if cnt is not None else name)
        authors_map[cid] = {"authors_formatted": formatted, "max_citation": max_cite}

    # 5) 构建 citing paper 元信息映射（用于文本输出的标题/作者(含计数)/期刊）
    citing_meta: Dict[str, Dict[str, Any]] = {}
    for it in items:
        cp = (it.get("citingPaper") or {})
        cid = cp.get("paperId")
        if not cid:
            continue
        title = cp.get("title") or ""
        venue = cp.get("venue") or ""
        a_formatted = authors_map.get(cid, {}).get("authors_formatted") or (cp.get("authors") or [])
        if isinstance(a_formatted, list):
            authors_str = "; ".join([str(a) for a in a_formatted])
        else:
            authors_str = str(a_formatted)
        max_cite = authors_map.get(cid, {}).get("max_citation") or 0
        citing_meta[cid] = {"title": title, "authors_str": authors_str, "venue": venue, "max_author_citation": max_cite}

    items_by_cid: Dict[str, Dict[str, Any]] = {}
    items_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, it in enumerate(items):
        it["refineIndex"] = idx
        items_by_idx[idx] = it
        cid = ((it.get("citingPaper") or {}).get("paperId"))
        if cid:
            items_by_cid[cid] = it

    def infer_citation_role(item: Dict[str, Any], rec: Optional[Dict[str, Any]] = None) -> str:
        ctxs = ((item.get("edge", {}) or {}).get("contexts") or [])[:3]
        intents = [str(x).lower() for x in (((item.get("edge", {}) or {}).get("intents") or []))]
        parts: List[str] = []
        if rec:
            for key in ("importance_reason", "category", "citation_role"):
                val = rec.get(key)
                if isinstance(val, str):
                    parts.append(val.lower())
            for s in rec.get("matched_keywords") or []:
                if isinstance(s, str):
                    parts.append(s.lower())
        for s in ctxs:
            if isinstance(s, str):
                parts.append(s.lower())
        parts.extend(intents)
        text = " ".join(parts)
        if any(k in text for k in ("based on", "build on", "built on", "inspired by", "adopt", "adopts", "adopted", "use ", "using ", "borrow")):
            return "方法借鉴"
        if any(k in text for k in ("baseline", "compare", "compared", "comparison", "against", "versus", "outperform", "outperformed")):
            return "性能比较"
        if any(k in text for k in ("dataset", "benchmark", "annotation", "annotated", "resource", "code", "corpus")):
            return "数据/资源引用"
        if any(k in text for k in ("representative", "remarkable progress", "significant improvement", "state-of-the-art", "landmark")):
            return "代表性工作"
        if any(k in text for k in ("application", "applied to", "transfer", "downstream", "extension", "extend")):
            return "扩展应用"
        if ctxs:
            return "背景铺垫"
        return "弱相关"

    def normalize_record(rec: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
        cp = item.get("citingPaper") or {}
        contexts = ((item.get("edge", {}) or {}).get("contexts") or [])[:3]
        normalized = dict(rec)
        normalized["refine_index"] = item.get("refineIndex")
        normalized["citing_paper_id"] = normalized.get("citing_paper_id") or cp.get("paperId")
        normalized["title"] = normalized.get("title") or cp.get("title")
        normalized["authors"] = normalized.get("authors") or cp.get("authors") or []
        normalized["year"] = normalized.get("year") or cp.get("year")
        normalized["venue"] = normalized.get("venue") or cp.get("venue")
        normalized["is_influential"] = bool(normalized.get("is_influential", (item.get("edge", {}) or {}).get("isInfluential")))
        normalized["intents"] = normalized.get("intents") or ((item.get("edge", {}) or {}).get("intents") or [])
        normalized["key_contexts"] = normalized.get("key_contexts") or contexts
        normalized["citation_role"] = normalized.get("citation_role") or infer_citation_role(item, normalized)
        return normalized

    def resolve_item_for_record(rec: Dict[str, Any], batch_scope: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        idx = rec.get("refine_index")
        if isinstance(idx, int):
            return items_by_idx.get(idx)
        idx2 = rec.get("i")
        if isinstance(idx2, int):
            rec["refine_index"] = idx2
            return items_by_idx.get(idx2)

        cid = rec.get("citing_paper_id")
        if isinstance(cid, str) and cid:
            return items_by_cid.get(cid)

        scope = batch_scope or items
        title_key = _normalize_title_key(rec.get("title"))
        year = rec.get("year")
        candidates: List[Dict[str, Any]] = []
        if title_key:
            for item in scope:
                item_title_key = _normalize_title_key(((item.get("citingPaper") or {}).get("title")))
                if item_title_key == title_key:
                    candidates.append(item)
        if len(candidates) == 1:
            matched = candidates[0]
            rec["citing_paper_id"] = ((matched.get("citingPaper") or {}).get("paperId"))
            return matched
        if len(candidates) > 1 and year is not None:
            for item in candidates:
                if ((item.get("citingPaper") or {}).get("year")) == year:
                    rec["citing_paper_id"] = ((item.get("citingPaper") or {}).get("paperId"))
                    return item

        if batch_scope and len(batch_scope) == 1:
            rec["citing_paper_id"] = (((batch_scope[0].get("citingPaper") or {}).get("paperId")))
            return batch_scope[0]
        return None

    out_jsonl, out_txt = citations_ref_paths_for_paper(cfg, paper_id)
    written = 0

    with out_jsonl.open("w", encoding="utf-8") as fj, out_txt.open("w", encoding="utf-8") as ft:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            # 将作者影响力信息注入到 batch，以便提示词考虑
            batch_enriched: List[Dict[str, Any]] = []
            for item in batch:
                cp = (item.get("citingPaper") or {})
                cid = cp.get("paperId")
                enriched = dict(item)
                enriched["refineIndex"] = item.get("refineIndex")
                enriched["authorsWithCitations"] = authors_map.get(cid or "", {}).get("authors_formatted", [])
                enriched["maxAuthorCitation"] = authors_map.get(cid or "", {}).get("max_citation", 0)
                batch_enriched.append(enriched)
            messages = build_refine_prompt(target_meta, batch_enriched)
            content = client.chat(messages, temperature=temperature)
            if not content:
                # LLM 无内容返回，直接对本批次使用启发式兜底，仍然产生输出，避免 0 条
                for item in batch:
                    intents = (item.get("edge", {}) or {}).get("intents") or []
                    is_inf = bool((item.get("edge", {}) or {}).get("isInfluential"))
                    intents_l = [str(i).lower() for i in intents]
                    if is_inf:
                        score = 5
                    elif any(k in intents_l for k in ("method", "methods", "approach", "technique")):
                        score = 4
                    elif any(k in intents_l for k in ("result", "results", "finding")):
                        score = 3
                    elif any(k in intents_l for k in ("background", "motivation")):
                        score = 2
                    elif any(k in intents_l for k in ("comparison", "extend", "extension")):
                        score = 3
                    else:
                        score = 1
                    reason = (
                        "根据 isInfluential 信号判定为关键引用"
                        if is_inf
                        else ("根据 intents 判定为较重要引用" if intents else "缺乏明确证据，保守判定为一般引用")
                    )
                    rec = {
                        "refine_index": item.get("refineIndex"),
                        "citing_paper_id": item["citingPaper"]["paperId"],
                        "importance_score": score,
                        "importance_reason": reason,
                        "category": (
                            "方法"
                            if any("method" in t for t in intents_l)
                            else "结果"
                            if any("result" in t for t in intents_l)
                            else "背景"
                            if any("background" in t for t in intents_l)
                            else "比较"
                            if any("comparison" in t for t in intents_l)
                            else "扩展"
                            if any("extend" in t or "extension" in t for t in intents_l)
                            else "其他"
                        ),
                        "citation_role": infer_citation_role(item),
                        "is_influential": is_inf,
                        "intents": intents,
                        "key_contexts": ((item.get("edge", {}) or {}).get("contexts") or [])[:3],
                    }
                    rec = normalize_record(rec, item)
                    fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cid_local = (item.get("citingPaper") or {}).get("paperId")
                    meta_local = citing_meta.get(cid_local or "", {})
                    header = "{} | {} | {}".format(
                        meta_local.get("title", "") or "",
                        meta_local.get("authors_str", "") or "",
                        meta_local.get("venue", "") or "",
                    )
                    tline = [
                        tsv_safe(header),
                        tsv_safe(rec.get("importance_score")),
                        tsv_safe(rec.get("citation_role")),
                        tsv_safe(rec.get("category")),
                        tsv_safe(rec.get("is_influential")),
                        tsv_safe(rec.get("importance_reason")),
                        tsv_safe(" || ".join(rec.get("key_contexts") or [])),
                        tsv_safe(f"MaxAuthorCitation: {meta_local.get('max_author_citation') or 0}"),
                    ]
                    ft.write("\t".join(tline) + "\n")
                    written += 1
                time.sleep(0.5)
                continue

            if echo_stdout:
                logging.info(f"LLM 原始输出（{paper_id} 批 {i // batch_size + 1}，{len(batch)}条）:\n{content}")
            try:
                raw_path = LOGS_DIR / f"llm_raw_{paper_id}_{i}_{len(batch)}.txt"
                raw_path.write_text(content, encoding="utf-8")
            except Exception:
                pass

            # 解析 LLM 输出：支持 NDJSON、代码块、或 JSON 数组
            text = content.strip()
            # 若包含代码块，提取代码块内部
            try:
                blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
                if blocks:
                    text = "\n".join([b.strip() for b in blocks if b])
            except Exception:
                pass

            records: List[Dict[str, Any]] = []

            # 尝试整体解析为 JSON（数组或对象）
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    for rec in parsed:
                        if isinstance(rec, dict):
                            records.append(rec)
                elif isinstance(parsed, dict):
                    records.append(parsed)
            except Exception:
                pass

            # 若仍为空，则按行解析 NDJSON，忽略数组符号与代码块标记
            if not records:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("```") or line.lower().startswith("json"):
                        continue
                    if line in ("[", "]", ","):
                        continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            records.append(rec)
                    except Exception:
                        continue

            # 若解析记录为空或条数不足，则按启发式生成兜底记录，确保每条输入都有输出
            existing_ids = set()
            for rec in records:
                if "s" in rec and "importance_score" not in rec:
                    rec["importance_score"] = rec.get("s")
                if "c" in rec and "category" not in rec:
                    rec["category"] = rec.get("c")
                if "r" in rec and "citation_role" not in rec:
                    rec["citation_role"] = rec.get("r")
                if "w" in rec and "importance_reason" not in rec:
                    rec["importance_reason"] = rec.get("w")
                if "i" in rec and "refine_index" not in rec:
                    rec["refine_index"] = rec.get("i")
                resolve_item_for_record(rec, batch)
                cid = rec.get("citing_paper_id")
                if isinstance(cid, str):
                    existing_ids.add(cid)

            def heuristic_record(item: Dict[str, Any]) -> Dict[str, Any]:
                cid = item["citingPaper"]["paperId"]
                intents = item.get("edge", {}).get("intents") or []
                is_inf = bool(item.get("edge", {}).get("isInfluential"))
                # 评分启发
                score = (
                    5
                    if is_inf
                    else (
                        3
                        if any(
                            x.lower() in ("method", "result", "extension", "background", "comparison")
                            for x in [str(i).lower() for i in intents]
                        )
                        else (2 if intents else 1)
                    )
                )
                reason = (
                    "根据 isInfluential 信号判定为关键引用"
                    if is_inf
                    else ("根据 intents 推测为较重要引用" if intents else "缺乏明确证据，保守判定为一般引用")
                )
                return {
                    "citing_paper_id": cid,
                    "importance_score": score,
                    "importance_reason": reason,
                    "category": "方法"
                    if any("method" in str(i).lower() for i in intents)
                    else (
                        "结果"
                        if any("result" in str(i).lower() for i in intents)
                        else (
                            "背景"
                            if any("background" in str(i).lower() for i in intents)
                            else ("比较" if any("comparison" in str(i).lower() for i in intents) else "其他")
                        )
                    ),
                    "citation_role": infer_citation_role(item),
                    "is_influential": is_inf,
                    "intents": intents,
                    "key_contexts": (item.get("edge", {}).get("contexts") or [])[:3],
                }

            # 填充缺失条目
            if len(records) < len(batch):
                for item in batch:
                    cid = item["citingPaper"]["paperId"]
                    if cid not in existing_ids:
                        records.append(heuristic_record(item))

            # 写入解析出的记录
            for rec in records:
                item = resolve_item_for_record(rec, batch)
                if item is None:
                    continue
                cid = rec.get("citing_paper_id")
                rec = normalize_record(rec, item)
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                meta = citing_meta.get(cid or "", {})
                header = "{} | {} | {}".format(
                    meta.get("title", "") or "",
                    meta.get("authors_str", "") or "",
                    meta.get("venue", "") or "",
                )
                tline = [
                    tsv_safe(header),
                    tsv_safe(rec.get("importance_score")),
                    tsv_safe(rec.get("citation_role")),
                    tsv_safe(rec.get("category")),
                    tsv_safe(rec.get("is_influential")),
                    tsv_safe(rec.get("importance_reason")),
                    tsv_safe(" || ".join(rec.get("key_contexts") or [])),
                    tsv_safe(f"MaxAuthorCitation: {meta.get('max_author_citation') or 0}"),
                ]
                ft.write("\t".join(tline) + "\n")
                written += 1

            time.sleep(0.5)

    logging.info(f"{paper_id} 精排结果: {written} 条")


def refine_citations(cfg: Dict[str, Any], only_first_n: Optional[int] = None) -> None:
    """
    批量精排 citations，行为保持与原 collect.py 一致
    """
    setup_dirs()
    # 收集待处理 paperId
    pub_jsonl = active_publications_jsonl()
    if not pub_jsonl.exists():
        logging.error("缺少 publications.jsonl，请先运行 fetch-publications")
        import sys  # local import to avoid top-level sys if unused elsewhere
        sys.exit(1)

    paper_ids: List[str] = []
    with pub_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            paper_ids.append(rec.get("paperId"))

    if only_first_n:
        paper_ids = paper_ids[:only_first_n]

    for pid in paper_ids:
        refine_citations_for_paper(cfg, pid)
