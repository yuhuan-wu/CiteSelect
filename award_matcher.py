#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import ROOT


DATASET_LABELS: Dict[str, str] = {
    "aaai-fellows.json": "AAAI Fellow",
    "aaas-fellows.json": "AAAS Fellow",
    "acm-dissertation-award.json": "ACM Dissertation Award",
    "acm-distinguished-member.json": "ACM Distinguished Member",
    "acm-fellow.json": "ACM Fellow",
    "acm-gordon-bell-prize.json": "ACM Gordon Bell Prize",
    "acm-grace-murray-hopper-award.json": "ACM Grace Murray Hopper Award",
    "acm-senior-member.json": "ACM Senior Member",
    "acm-turing-award.json": "ACM Turing Award",
    "amacad-members.json": "American Academy of Arts and Sciences Member",
    "ieee-fellows.json": "IEEE Fellow",
    "nas-members.json": "NAS Member",
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _clean_name_text(name: str) -> str:
    text = _strip_accents(name or "")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\((.*?)\)", r" \1 ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Za-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _tokenize(name: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9']+", _clean_name_text(name)) if tok]


def _compact_tokens(tokens: List[str]) -> List[str]:
    compact = [tok for tok in tokens if len(tok) > 1]
    return compact or tokens


def _name_variants(name: str) -> List[str]:
    raw = (name or "").strip()
    if not raw:
        return []
    variants = {raw}
    no_paren = re.sub(r"\([^)]*\)", " ", raw)
    no_paren = re.sub(r"\s+", " ", no_paren).strip()
    if no_paren:
        variants.add(no_paren)
    keep_inner = re.sub(r"[()]", " ", raw)
    keep_inner = re.sub(r"\s+", " ", keep_inner).strip()
    if keep_inner:
        variants.add(keep_inner)
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 2:
            variants.add(f"{parts[1]} {parts[0]}")
    return [v for v in variants if v]


def _initials(tokens: List[str]) -> str:
    return "".join(tok[0] for tok in tokens if tok)


def _token_key(tokens: List[str]) -> str:
    return " ".join(tokens)


def _year_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(\d{4})", str(value))
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _similarity(query_tokens: List[str], cand_tokens: List[str]) -> float:
    query_key = _token_key(query_tokens)
    cand_key = _token_key(cand_tokens)
    if query_key == cand_key:
        return 1.0

    query_compact = _compact_tokens(query_tokens)
    cand_compact = _compact_tokens(cand_tokens)
    if _token_key(query_compact) == _token_key(cand_compact):
        return 0.99

    if query_compact and cand_compact and query_compact[-1] == cand_compact[-1]:
        q_given = query_compact[:-1]
        c_given = cand_compact[:-1]
        if q_given and c_given:
            q_counter = Counter(q_given)
            c_counter = Counter(c_given)
            if q_counter == c_counter:
                return 0.98
            if len(q_given) == len(c_given):
                compatible = True
                for q_tok, c_tok in zip(q_given, c_given):
                    if q_tok == c_tok:
                        continue
                    if len(q_tok) == 1 and c_tok.startswith(q_tok):
                        continue
                    if len(c_tok) == 1 and q_tok.startswith(c_tok):
                        continue
                    compatible = False
                    break
                if compatible:
                    return 0.96

    return SequenceMatcher(None, query_key, cand_key).ratio()


class AcademicAwardsMatcher:
    def __init__(
        self,
        data_dir: Path,
        *,
        fuzzy_threshold: float = 0.94,
        max_matches_per_author: int = 3,
    ) -> None:
        self.data_dir = data_dir
        self.fuzzy_threshold = max(0.8, min(0.99, float(fuzzy_threshold)))
        self.max_matches_per_author = max(1, int(max_matches_per_author))
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._exact_index: Dict[str, List[Dict[str, Any]]] = {}
        self._compact_index: Dict[str, List[Dict[str, Any]]] = {}
        self._last_name_index: Dict[str, List[Dict[str, Any]]] = {}
        self._load_entries()

    def _add_index(self, index: Dict[str, List[Dict[str, Any]]], key: str, entry: Dict[str, Any]) -> None:
        if not key:
            return
        bucket = index.setdefault(key, [])
        bucket.append(entry)

    def _load_entries(self) -> None:
        files = sorted(self.data_dir.glob("*.json"))
        for path in files:
            label = DATASET_LABELS.get(path.name)
            if not label:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                name = str(rec.get("name") or "").strip()
                if not name:
                    continue
                variants = _name_variants(name)
                token_variants: List[List[str]] = []
                compact_variants: List[List[str]] = []
                for variant in variants:
                    tokens = _tokenize(variant)
                    if not tokens:
                        continue
                    token_variants.append(tokens)
                    compact_variants.append(_compact_tokens(tokens))
                if not token_variants:
                    continue
                entry = {
                    "award": label,
                    "award_source": path.name,
                    "name": name,
                    "year": _year_value(rec.get("year")),
                    "token_variants": token_variants,
                    "compact_variants": compact_variants,
                    "last_names": sorted({tokens[-1] for tokens in compact_variants if tokens}),
                    "first_initials": sorted({_initials(tokens[:1]) for tokens in compact_variants if tokens}),
                }
                self._entries.append(entry)
                for tokens in token_variants:
                    self._add_index(self._exact_index, _token_key(tokens), entry)
                for tokens in compact_variants:
                    self._add_index(self._compact_index, _token_key(tokens), entry)
                    if tokens:
                        self._add_index(self._last_name_index, tokens[-1], entry)

    def _dedupe_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for match in sorted(matches, key=lambda x: (-float(x.get("score", 0.0)), str(x.get("award")), str(x.get("matched_name")))):
            key = (match.get("award"), match.get("matched_name"))
            if key in seen:
                continue
            seen.add(key)
            out.append(match)
            if len(out) >= self.max_matches_per_author:
                break
        return out

    def _match_name_uncached(self, name: str) -> List[Dict[str, Any]]:
        variants = _name_variants(name)
        if not variants:
            return []

        exact_candidates: List[Dict[str, Any]] = []
        query_token_variants: List[List[str]] = []
        query_compact_variants: List[List[str]] = []
        for variant in variants:
            tokens = _tokenize(variant)
            if not tokens:
                continue
            compact = _compact_tokens(tokens)
            query_token_variants.append(tokens)
            query_compact_variants.append(compact)
            exact_candidates.extend(self._exact_index.get(_token_key(tokens), []))
            exact_candidates.extend(self._compact_index.get(_token_key(compact), []))

        if exact_candidates:
            return self._dedupe_matches([
                {
                    "award": entry["award"],
                    "matched_name": entry["name"],
                    "award_year": entry.get("year"),
                    "award_source": entry["award_source"],
                    "score": 1.0,
                }
                for entry in exact_candidates
            ])

        fuzzy_candidates: List[Dict[str, Any]] = []
        for compact in query_compact_variants:
            if len(compact) < 2:
                continue
            last_name = compact[-1]
            first_initial = _initials(compact[:1])
            candidates = self._last_name_index.get(last_name, [])
            for entry in candidates:
                if first_initial and entry.get("first_initials") and first_initial not in entry["first_initials"]:
                    continue
                best_score = 0.0
                for cand_tokens in entry.get("compact_variants", []):
                    if not cand_tokens or cand_tokens[-1] != last_name:
                        continue
                    score = _similarity(compact, cand_tokens)
                    if score > best_score:
                        best_score = score
                if best_score >= self.fuzzy_threshold:
                    fuzzy_candidates.append({
                        "award": entry["award"],
                        "matched_name": entry["name"],
                        "award_year": entry.get("year"),
                        "award_source": entry["award_source"],
                        "score": round(best_score, 4),
                    })
        return self._dedupe_matches(fuzzy_candidates)

    def match_name(self, name: str) -> List[Dict[str, Any]]:
        key = _clean_name_text(name)
        if not key:
            return []
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        matches = self._match_name_uncached(name)
        with self._cache_lock:
            self._cache[key] = list(matches)
        return list(matches)

    def match_names(self, names: List[str], max_workers: int = 8) -> Dict[str, List[Dict[str, Any]]]:
        unique_names = [name for name in dict.fromkeys([str(name or "").strip() for name in names]) if name]
        if not unique_names:
            return {}
        if max_workers <= 1:
            return {name: self.match_name(name) for name in unique_names}

        results: Dict[str, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(self.match_name, name): name for name in unique_names}
            for future, name in future_map.items():
                results[name] = future.result()
        return results


_MATCHER_SINGLETON: Optional[AcademicAwardsMatcher] = None
_MATCHER_LOCK = threading.Lock()
_MATCHER_KEY: Optional[Tuple[str, float, int]] = None


def get_awards_matcher(cfg: Dict[str, Any]) -> AcademicAwardsMatcher:
    awards_cfg = cfg.get("awards") or {}
    data_dir_raw = str(awards_cfg.get("data_dir") or "academic-awards/data")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    fuzzy_threshold = float(awards_cfg.get("fuzzy_threshold", 0.94))
    max_matches_per_author = int(awards_cfg.get("max_matches_per_author", 3))
    key = (str(data_dir.resolve()), fuzzy_threshold, max_matches_per_author)

    global _MATCHER_SINGLETON, _MATCHER_KEY
    with _MATCHER_LOCK:
        if _MATCHER_SINGLETON is None or _MATCHER_KEY != key:
            _MATCHER_SINGLETON = AcademicAwardsMatcher(
                data_dir=data_dir,
                fuzzy_threshold=fuzzy_threshold,
                max_matches_per_author=max_matches_per_author,
            )
            _MATCHER_KEY = key
        return _MATCHER_SINGLETON
