#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from typing import List, Dict, Any, Optional

from utils import (
    # setup/config/state
    setup_logging,
    load_config,
    setup_dirs,
    load_state,
    save_state,
    # constants and helpers
    PUB_TXT,
    PUB_JSONL,
    tsv_safe,
    author_names,
    paper_doi,
    match_pubtype,
    parse_year_range,
    build_semanticscholar,
)

from fetch_citations import fetch_citations
from refine_citations import refine_citations
from export_excel import export_excel


def fetch_publications(cfg: Dict[str, Any]) -> None:
    """
    抓取作者论文列表，输出 publications.jsonl 与 publications.txt
    行为与原实现保持一致
    """
    setup_dirs()
    s2 = build_semanticscholar(cfg)
    author_id = cfg.get("author_id")
    if not author_id:
        logging.error("author_id 未设置；请在 config.yaml 的 author_url 或 author_id 中填写")
        sys.exit(1)

    yr_lo, yr_hi = parse_year_range(cfg.get("year_range", ""))
    include_types = cfg.get("include_types") or []
    limit = int(cfg.get("page_size", 100))

    logging.info(f"开始抓取作者 {author_id} 的论文列表...")
    results = s2.get_author_papers(author_id=author_id, fields=None, limit=limit)

    papers = []
    for p in results:
        y = getattr(p, "year", None)
        if yr_lo and yr_hi:
            if not y or y < yr_lo or y > yr_hi:
                continue
        if not match_pubtype(getattr(p, "publicationTypes", None), include_types):
            continue
        papers.append(p)

    papers.sort(key=lambda p: getattr(p, "citationCount", 0) or 0, reverse=True)
    logging.info(f"过滤后论文数: {len(papers)}")

    # write publications.jsonl and publications.txt
    if papers:
        with PUB_JSONL.open("w", encoding="utf-8") as fjson, PUB_TXT.open("w", encoding="utf-8") as ftxt:
            for p in papers:
                rec = {
                    "paperId": p.paperId,
                    "title": p.title,
                    "year": p.year,
                    "venue": p.venue,
                    "publicationTypes": getattr(p, "publicationTypes", None),
                    "citationCount": p.citationCount,
                    "url": p.url,
                    "externalIds": getattr(p, "externalIds", None),
                    "authors": author_names(p),
                    "raw": p.raw_data,
                }
                fjson.write(json.dumps(rec, ensure_ascii=False) + "\n")
                line = [
                    tsv_safe(p.paperId),
                    tsv_safe(p.year),
                    tsv_safe(p.citationCount),
                    tsv_safe(p.title),
                    tsv_safe("; ".join(author_names(p))),
                    tsv_safe(p.venue),
                    tsv_safe("|".join(getattr(p, "publicationTypes", []) or [])),
                    tsv_safe(p.url),
                    tsv_safe(paper_doi(p) or ""),
                ]
                ftxt.write("\t".join(line) + "\n")

    # 保存状态（用于续跑）
    try:
        state = load_state()
        state["author_id"] = author_id
        state["publications_done"] = True
        state["paper_ids"] = [getattr(p, "paperId", None) for p in papers]
        save_state(state)
    except Exception as e:
        logging.warning(f"保存 state 失败（非致命）：{e}")

    logging.info(f"已写入 {PUB_TXT} 与 {PUB_JSONL}")


def cmd_all(cfg: Dict[str, Any]) -> None:
    fetch_publications(cfg)
    fetch_citations(cfg)
    refine_citations(cfg)
    export_excel(cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic Scholar 关键引用抓取与精排工具")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fetch-publications")
    pc = sp.add_parser("fetch-citations")
    pc.add_argument("--first-n", type=int, default=None, help="仅处理前 N 篇作者论文（调试用）")
    pc.add_argument("--per-paper-max", type=int, default=None, help="每篇论文最多抓取的引用数（调试用）")
    pr = sp.add_parser("refine-citations")
    pr.add_argument("--first-n", type=int, default=None, help="仅处理前 N 篇作者论文（调试用）")
    pr.add_argument("--echo-llm", dest="echo_llm", action="store_true", help="打印LLM原始输出到控制台")
    pr.add_argument("--no-echo-llm", dest="echo_llm", action="store_false", help="不打印LLM原始输出到控制台")
    pr.set_defaults(echo_llm=None)
    pe = sp.add_parser("export-excel")
    pe.add_argument("--first-n", type=int, default=None, help="仅导出前 N 篇作者论文（调试用）")
    sp.add_parser("all")
    return p


def main():
    setup_logging()
    cfg = load_config()
    args = build_arg_parser().parse_args()
    if args.cmd == "fetch-publications":
        fetch_publications(cfg)
    elif args.cmd == "fetch-citations":
        fetch_citations(
            cfg,
            only_first_n=getattr(args, "first_n", None),
            per_paper_max=getattr(args, "per_paper_max", None),
        )
    elif args.cmd == "refine-citations":
        if getattr(args, "echo_llm", None) is not None:
            cfg.setdefault("llm", {})
            cfg["llm"]["echo_stdout"] = bool(args.echo_llm)
        refine_citations(cfg, only_first_n=getattr(args, "first_n", None))
    elif args.cmd == "export-excel":
        export_excel(cfg, only_first_n=getattr(args, "first_n", None))
    elif args.cmd == "all":
        cmd_all(cfg)


if __name__ == "__main__":
    main()
