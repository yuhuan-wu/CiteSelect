#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from semanticscholar import SemanticScholar


# -----------------------------
# Paths and Constants
# -----------------------------
ROOT: Path = Path(__file__).parent.resolve()
OUTPUTS_DIR: Path = ROOT / "outputs"
PUB_DIR: Path = OUTPUTS_DIR / "publications"
CITES_DIR: Path = OUTPUTS_DIR / "citations"
CITES_REF_DIR: Path = OUTPUTS_DIR / "citations_refined"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
CACHE_DIR: Path = OUTPUTS_DIR / "cache"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
PUB_TXT: Path = PUB_DIR / "publications.txt"
PUB_JSONL: Path = PUB_DIR / "publications.jsonl"
LEGACY_PUB_TXT: Path = ROOT / "publications.txt"
LEGACY_PUB_JSONL: Path = ROOT / "publications.jsonl"
LEGACY_CITES_DIR: Path = ROOT / "citations"
LEGACY_CITES_REF_DIR: Path = ROOT / "citations_refined"
LEGACY_CACHE_DIR: Path = ROOT / "cache"
LEGACY_LOGS_DIR: Path = ROOT / "logs"
STATE_FILE: Path = CACHE_DIR / "state.json"
CONFIG_FILE: Path = ROOT / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "author_url": "",
    "author_id": "",
    "s2_api_key": "USBpwJFyAp1fJWr3VkLqn6wbWkD0HTxJ42q5o0a3",
    "openai_api_key": None,
    "openrouter_api_key": None,
    "year_range": "2018-2025",
    "include_types": ["journal", "conference", "preprint"],
    "page_size": 100,
    "max_concurrency": 3,
    "s2_qps": 1,
    "retry": 3,
    "output": {
        "file_naming": "title",
        "write_jsonl": True,
        "write_txt": True,
        "resume": True,
        "force": False,
    },
    "llm": {
        "provider": "openrouter",
        "model": "openai/gpt-5.4",
        "base_url": "https://openrouter.ai/api/v1",
        "app_name": "CiteSelect",
        "site_url": "https://github.com/",
        "batch_size": 40,
        "temperature": 1.0,
        "max_tokens": 1024,
        "cost_cap_usd": None,
        "echo_stdout": True,
    },
    "awards": {
        "enabled": True,
        "data_dir": "academic-awards/data",
        "max_workers": 8,
        "fuzzy_threshold": 0.94,
        "max_matches_per_author": 3,
        "llm_filter": {
            "enabled": False,
            "batch_size": 40,
            "max_output_tokens": 1200,
            "cache_path": "outputs/cache/award_llm_filter_cache.json",
        },
    },
    "export": {
        "output_path": "outputs/reports/representative_citations.xlsx",
        "top_score_threshold": 4,
        "author_profiles_path": "author_profiles.yaml",
        "venue_llm_filter": {
            "enabled": True,
            "batch_size": 50,
            "max_output_tokens": 1200,
            "cache_path": "outputs/cache/venue_llm_filter_cache.json",
        },
    },
}


# -----------------------------
# Utilities: filesystem, logging, yaml/config/state
# -----------------------------
def setup_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    CITES_DIR.mkdir(parents=True, exist_ok=True)
    CITES_REF_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    setup_dirs()
    log_path = LOGS_DIR / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    if path.exists() and yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_year_range(s: str) -> Tuple[Optional[int], Optional[int]]:
    if not s:
        return None, None
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_author_id(author_url: str) -> Optional[str]:
    if not author_url:
        return None
    m = re.search(r"/author/[^/]+/(\d+)", author_url)
    return m.group(1) if m else None


def load_state() -> Dict[str, Any]:
    state_path = STATE_FILE if STATE_FILE.exists() else (LEGACY_CACHE_DIR / "state.json")
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    file_cfg = load_yaml(CONFIG_FILE)

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> None:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                deep_merge(a[k], v)  # type: ignore[index]
            else:
                a[k] = v

    deep_merge(cfg, file_cfg)
    # env fallback
    cfg["s2_api_key"] = cfg.get("s2_api_key") or os.environ.get("S2_API_KEY")
    cfg["openai_api_key"] = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    cfg["openrouter_api_key"] = cfg.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY")
    # author id
    if not cfg.get("author_id"):
        cfg["author_id"] = parse_author_id(cfg.get("author_url", "")) or ""
    return cfg


# -----------------------------
# Data helpers
# -----------------------------
def tsv_safe(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return s


def author_names(paper_obj) -> List[str]:
    try:
        return [a.name for a in (paper_obj.authors or []) if getattr(a, "name", None)]
    except Exception:
        return []


def paper_doi(paper_obj) -> Optional[str]:
    try:
        ex = paper_obj.externalIds or {}
        return ex.get("DOI")
    except Exception:
        return None


def match_pubtype(types: Any, allowed: List[str]) -> bool:
    if not types:
        return True  # if missing, keep
    allowed_l = [t.lower() for t in allowed]
    if isinstance(types, list):
        vals = [str(t).lower() for t in types]
    else:
        vals = [str(types).lower()]
    for v in vals:
        for a in allowed_l:
            if a in v:
                return True
    return False


# -----------------------------
# Clients / Rate limiting
# -----------------------------
class RateLimiter:
    def __init__(self, qps: float):
        self.qps = max(0.1, float(qps))
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.time()
            if now < self.next_allowed:
                time.sleep(self.next_allowed - now)
                now = time.time()
            self.next_allowed = now + 1.0 / self.qps


def build_semanticscholar(cfg: Dict[str, Any]) -> SemanticScholar:
    return SemanticScholar(api_key=cfg.get("s2_api_key"), timeout=30, retry=True)


# -----------------------------
# Paths helpers
# -----------------------------
def active_publications_jsonl() -> Path:
    if PUB_JSONL.exists():
        return PUB_JSONL
    return LEGACY_PUB_JSONL


def active_publications_txt() -> Path:
    if PUB_TXT.exists():
        return PUB_TXT
    return LEGACY_PUB_TXT


def citations_paths(paper_id: str) -> Tuple[Path, Path]:
    return (CITES_DIR / f"{paper_id}.jsonl", CITES_DIR / f"{paper_id}.txt")


def citations_ref_paths(paper_id: str) -> Tuple[Path, Path]:
    return (CITES_REF_DIR / f"{paper_id}.jsonl", CITES_REF_DIR / f"{paper_id}.txt")


def legacy_citations_paths(paper_id: str) -> Tuple[Path, Path]:
    return (LEGACY_CITES_DIR / f"{paper_id}.jsonl", LEGACY_CITES_DIR / f"{paper_id}.txt")


def legacy_citations_ref_paths(paper_id: str) -> Tuple[Path, Path]:
    return (LEGACY_CITES_REF_DIR / f"{paper_id}.jsonl", LEGACY_CITES_REF_DIR / f"{paper_id}.txt")


def title_to_safe_filename(title: str) -> str:
    """
    将论文标题转换为适合作为文件名的字符串：
    - 删除常见非法字符：<>:"/\|?* 以及控制字符
    - 保留中英文、数字、空格、常用符号（去除零宽字符）
    - 折叠多余空白并裁剪长度（默认120）
    """
    s = title or ""
    # 删除常见非法字符和控制字符
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", s)
    # 删除零宽字符
    s = s.replace("\u200b", "")
    # 折叠空白
    s = re.sub(r"\s+", " ", s).strip()
    # 去除末尾的点和空格
    s = s.strip(" .")
    # 限制长度
    if len(s) > 120:
        s = s[:120].rstrip()
    return s or "untitled"


def citations_paths_for_paper(cfg: Dict[str, Any], paper_id: str) -> Tuple[Path, Path]:
    """
    根据配置 output.file_naming 计算 citations 的输出路径：
    - "title": 使用目标论文标题（经清洗）作为文件名
    - 其他值：回退到 paperId 命名
    """
    mode = ((cfg.get("output") or {}).get("file_naming") or "title").lower()
    if mode == "title":
        title: Optional[str] = None
        try:
            pub_jsonl = active_publications_jsonl()
            if pub_jsonl.exists():
                with pub_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("paperId") == paper_id:
                            title = rec.get("title")
                            break
        except Exception:
            title = None
        base = title_to_safe_filename(title or "")
        if not base:
            base = paper_id
        return (CITES_DIR / f"{base}.jsonl", CITES_DIR / f"{base}.txt")
    else:
        return citations_paths(paper_id)


def citations_ref_paths_for_paper(cfg: Dict[str, Any], paper_id: str) -> Tuple[Path, Path]:
    """
    根据配置 output.file_naming 计算 citations_refined 的输出路径：
    - "title": 使用目标论文标题（经清洗）作为文件名
    - 其他值：回退到 paperId 命名
    """
    mode = ((cfg.get("output") or {}).get("file_naming") or "title").lower()
    if mode == "title":
        title: Optional[str] = None
        try:
            pub_jsonl = active_publications_jsonl()
            if pub_jsonl.exists():
                with pub_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("paperId") == paper_id:
                            title = rec.get("title")
                            break
        except Exception:
            title = None
        base = title_to_safe_filename(title or "")
        if not base:
            base = paper_id
        return (CITES_REF_DIR / f"{base}.jsonl", CITES_REF_DIR / f"{base}.txt")
    else:
        return citations_ref_paths(paper_id)


__all__ = [
    # paths
    "ROOT", "OUTPUTS_DIR", "PUB_DIR", "PUB_TXT", "PUB_JSONL", "LEGACY_PUB_TXT", "LEGACY_PUB_JSONL",
    "CITES_DIR", "CITES_REF_DIR", "REPORTS_DIR", "CACHE_DIR", "LOGS_DIR",
    "LEGACY_CITES_DIR", "LEGACY_CITES_REF_DIR", "LEGACY_CACHE_DIR", "LEGACY_LOGS_DIR",
    "STATE_FILE", "CONFIG_FILE",
    # config/defaults
    "DEFAULT_CONFIG", "load_yaml", "load_config",
    # setup
    "setup_dirs", "setup_logging",
    # parse/tools
    "parse_year_range", "parse_author_id", "tsv_safe", "author_names", "paper_doi", "match_pubtype",
    # state
    "load_state", "save_state",
    # clients/rate limit
    "RateLimiter", "build_semanticscholar",
    # paths helpers
    "active_publications_jsonl", "active_publications_txt",
    "citations_paths", "citations_ref_paths", "legacy_citations_paths", "legacy_citations_ref_paths",
    "title_to_safe_filename", "citations_paths_for_paper", "citations_ref_paths_for_paper",
]

