"""v5_optimized blocking: Multi-strategy candidate generation for >99.5% recall"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import Config, Record, load_source, load_truth
from jellyfish import metaphone as _metaphone

# Copied constants from normalize.py to avoid relative import issues
_LEGAL_TOKENS = {
    "corp", "corporation", "inc", "incorporated", "llc", "llp", "ltd", "limited",
    "co", "company", "cos", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "solutions", "technologies", "tech",
    "international", "internationl", "intl", "consulting", "associates",
    "trading", "traders", "industries", "agency", "agencies", "stores", "store",
    "and", "the", "of", "for", "usa", "us",
    "pvt", "private", "sons", "india",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "gie",
    "societe", "ets", "etablissements", "cie", "france", "fr",
}

_ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent", "next",
    "to", "the", "and", "at", "post", "po", "no", "room", "floor", "flr",
    "dist", "district", "taluk", "tehsil", "road", "rd", "street", "st",
    "lane", "ln", "avenue", "ave", "highway", "cross", "main", "phase",
    "sector", "sec", "block", "building", "bldg", "house", "flat", "flt",
    "shop", "gala", "complex", "area", "nagar", "colony", "enclave",
    "vihar", "layout", "city", "state", "pin", "code", "zip",
    "india", "us", "usa", "france",
    "rue", "chemin", "quai", "allee", "route", "cedex", "bp",
}

_ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "blv": "boulevard", "bd": "boulevard",
    "ln": "lane", "dr": "drive", "drv": "drive", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ct": "court", "plz": "plaza",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch",
    "rue": "street", "avenu": "avenue", "chem": "chemin", "quai": "quay",
    "all": "allee", "rte": "route", "imp": "impasse",
}

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def _to_ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFKC", s)


def _normalize_text(s: str) -> str:
    s = _to_ascii_fold(s or "").lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def _expand_abbrev(tokens: list[str]) -> list[str]:
    return [_ADDR_ABBREV.get(t, t) for t in tokens]


def _numbers_from(tokens: list[str]) -> tuple:
    out = []
    for t in tokens:
        for n in _NUM_RE.findall(t):
            if len(n) >= 2 or not tokens:
                out.append(n.lstrip("0") or "0")
    return tuple(sorted(set(out)))


def _pins_from(tokens: list[str]) -> frozenset:
    return frozenset(t for t in tokens if t.isdigit() and 5 <= len(t) <= 6)


def build_record(entity_id: str, name: str, address: str, country: str) -> Record:
    from common import Record
    r = Record(entity_id=entity_id, name=name, address=address,
               country=_normalize_text(country) or "unknown")
    name_toks = _normalize_text(name).split()
    addr_toks = _expand_abbrev(_normalize_text(address).split())

    r.name_tokens = frozenset(name_toks)
    r.core_tokens = frozenset(t for t in name_toks
                              if t not in _LEGAL_TOKENS and len(t) >= 2)
    r.addr_tokens = frozenset(t for t in addr_toks if t not in _ADDR_STOP and len(t) >= 2)
    r.numbers = _numbers_from(addr_toks)
    r.pins = _pins_from(addr_toks)
    r.name_norm = " ".join(name_toks)
    r.addr_norm = " ".join(addr_toks)
    core_sorted = sorted(r.core_tokens, key=lambda t: (-len(t), t))
    r.metaphone = _metaphone(core_sorted[0]) if core_sorted else ""
    r.name_prefix = r.name_norm[:3] if len(r.name_norm) >= 3 else r.name_norm
    return r


def _doc_freq(records: list[Record]) -> dict[str, int]:
    df: dict[str, int] = defaultdict(int)
    for rec in records:
        for tok in rec.core_tokens:
            df[tok] += 1
    return dict(df)


def _record_keys_v1(rec: Record, df_counts: dict[str, int], cfg: Config) -> list[tuple]:
    """Original key blocking strategy"""
    keys: list[tuple] = []
    for tok in rec.core_tokens:
        if len(tok) >= cfg.min_token_len and df_counts.get(tok, 0) <= cfg.rare_df_max:
            keys.append(("t", tok))
    for num in rec.numbers:
        keys.append(("n", num))
    for pin in rec.pins:
        keys.append(("pin", rec.country, pin))
    sig = " ".join(sorted(rec.core_tokens)[: cfg.sig_tokens])
    if sig:
        keys.append(("s", rec.country, sig))
    if rec.name_prefix:
        keys.append(("p", rec.country, rec.name_prefix))
    if rec.metaphone:
        keys.append(("m", rec.country, rec.metaphone))
    for atok in rec.addr_tokens:
        if len(atok) >= 5 and not atok.isdigit():
            keys.append(("a", rec.country, atok))
    return keys


def _record_keys_v2(rec: Record, df_counts: dict[str, int], cfg: Config) -> list[tuple]:
    """Enhanced key blocking: more key types, cross-country"""
    keys: list[tuple] = []
    
    # 1. Rare name tokens (country-agnostic + country-specific)
    for tok in rec.core_tokens:
        if len(tok) >= cfg.min_token_len:
            df = df_counts.get(tok, 0)
            if df <= cfg.rare_df_max:
                keys.append(("t", tok))           # global rare
                keys.append(("tc", rec.country, tok))  # country-specific rare
    
    # 2. All name tokens (not just rare) for high-recall
    for tok in rec.core_tokens:
        if len(tok) >= 4:
            keys.append(("n_all", tok))
    
    # 3. Address numbers (critical for India/US)
    for num in rec.numbers:
        if len(num) >= 2:
            keys.append(("num", num))
            keys.append(("num_c", rec.country, num))
    
    # 4. PIN/ZIP codes - very strong signal
    for pin in rec.pins:
        keys.append(("pin", pin))                 # global
        keys.append(("pin_c", rec.country, pin))  # country-specific
    
    # 5. Sorted signature (name tokens)
    sig = " ".join(sorted(rec.core_tokens)[: cfg.sig_tokens])
    if sig:
        keys.append(("sig", sig))
        keys.append(("sig_c", rec.country, sig))
    
    # 6. Prefix keys
    if rec.name_prefix:
        keys.append(("pref", rec.name_prefix))
        keys.append(("pref_c", rec.country, rec.name_prefix))
    
    # 7. Metaphone of longest token
    if rec.metaphone:
        keys.append(("mph", rec.metaphone))
        keys.append(("mph_c", rec.country, rec.metaphone))
    
    # 8. Address tokens (distinctive ones)
    for atok in rec.addr_tokens:
        if len(atok) >= 5 and not atok.isdigit():
            keys.append(("addr", atok))
            keys.append(("addr_c", rec.country, atok))
    
    # 9. First token of name (very strong)
    if rec.core_tokens:
        first = rec.core_tokens[0] if isinstance(rec.core_tokens, (list, tuple)) else next(iter(rec.core_tokens))
        if len(first) >= 3:
            keys.append(("first", first))
            keys.append(("first_c", rec.country, first))
    
    # 10. Bigrams of name tokens (catches transpositions)
    core_list = list(rec.core_tokens)
    for i in range(len(core_list) - 1):
        bg = core_list[i] + " " + core_list[i + 1]
        keys.append(("bg", bg))
    
    # 11. Country-only bucket (fallback)
    keys.append(("country", rec.country))
    
    return keys


class MultiStrategyBlocker:
    """Multiple blocking strategies UNIONED for maximum recall"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.strategies = [
            (_record_keys_v1, "v1_original"),
            (_record_keys_v2, "v2_enhanced"),
        ]
        self.indexes: dict[str, dict[tuple, list[int]]] = {}
        self.df_counts: dict[str, int] = {}
    
    def build_index(self, other_records: list[Record]):
        """Build inverted indexes for all strategies"""
        self.df_counts = _doc_freq(other_records)
        
        for key_fn, name in self.strategies:
            index: dict[tuple, list[int]] = defaultdict(list)
            for j, rec in enumerate(other_records):
                for key in key_fn(rec, self.df_counts, self.cfg):
                    index[key].append(j)
            # Drop oversized buckets
            self.indexes[name] = {
                k: v for k, v in index.items() 
                if len(v) <= self.cfg.max_bucket
            }
    
    def query(self, s1_records: list[Record]) -> dict[int, set[int]]:
        """Query all strategies and UNION results"""
        cands_idx: dict[int, set[int]] = defaultdict(set)
        
        for i, rec in enumerate(s1_records):
            for key_fn, name in self.strategies:
                index = self.indexes[name]
                for key in key_fn(rec, self.df_counts, self.cfg):
                    bucket = index.get(key)
                    if bucket:
                        cands_idx[i].update(bucket)
        
        return dict(cands_idx)


class VectorIndex:
    """Char n-gram TF-IDF cosine index for fuzzy blocking"""
    
    def __init__(self, s1_records: list[Record], other_records: list[Record], cfg: Config):
        texts = [f"{r.name_norm} {r.addr_norm}" for r in s1_records] + \
                [f"{r.name_norm} {r.addr_norm}" for r in other_records]
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(cfg.char_ngram_lo, cfg.char_ngram_hi),
            min_df=2,
            max_features=cfg.tfidf_max_features,
            dtype=np.float32,
            sublinear_tf=True,
        )
        M = vec.fit_transform(texts or [" "])
        self.M1 = M[: len(s1_records)].tocsr()
        self.M2 = M[len(s1_records):].tocsr()
        self.n_other = len(other_records)
        self.chunk = int(min(512, max(1, 32_000_000 // max(1, self.n_other))))
    
    def topk_bulk(self, k: int) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        if self.n_other == 0:
            return out
        for start in range(0, self.M1.shape[0], self.chunk):
            sims = (self.M1[start:start + self.chunk] @ self.M2.T).toarray()
            k_eff = min(k, sims.shape[1])
            idx = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
            for r in range(sims.shape[0]):
                i = start + r
                out[i] = {int(j) for j in idx[r] if sims[r, j] > 1e-6}
        return out
    
    def trim(self, i: int, cand: set[int], k: int) -> set[int]:
        if len(cand) <= k:
            return cand
        js = np.fromiter(cand, dtype=np.int64, count=len(cand))
        prod = self.M1[i] @ self.M2[js].T
        sims = np.asarray(prod.todense()).ravel()
        keep = np.argsort(-sims)[:k]
        return {int(js[j]) for j in keep}


def generate_candidates(
    s1_records: list[Record],
    other_records: list[Record],
    cfg: Config,
) -> tuple[dict[int, set[int]], dict[str, list[str]]]:
    """Multi-strategy blocking + vector layer UNION"""
    
    # 1. Key-based blocking (multiple strategies)
    blocker = MultiStrategyBlocker(cfg)
    blocker.build_index(other_records)
    cands_idx = blocker.query(s1_records)
    
    # 2. Vector (fuzzy) blocking layer
    if cfg.vector_topk > 0:
        vi = VectorIndex(s1_records, other_records, cfg)
        vec_cands = vi.topk_bulk(cfg.vector_topk)
        for i, js in vec_cands.items():
            cands_idx[i] = cands_idx.get(i, set()) | js
    
    # 3. Optional per-entity top-K trimming by cosine
    if cfg.per_entity_topk > 0 and cfg.vector_topk > 0:
        vi = VectorIndex(s1_records, other_records, cfg)
        for i in list(cands_idx):
            cands_idx[i] = vi.trim(i, cands_idx[i], cfg.per_entity_topk)
    
    # Convert to entity_id lists
    cands_id = {
        s1_records[i].entity_id: sorted(other_records[j].entity_id for j in js)
        for i, js in cands_idx.items()
    }
    
    return cands_idx, cands_id


def blocking_recall(
    cands_idx: dict[int, set[int]],
    s1_records: list[Record],
    other_by_id: dict[str, int],
    truth: dict[str, set[str]],
    entity_ids: set[str] | None = None
) -> tuple[float, int, int]:
    found, total = 0, 0
    for i, rec in enumerate(s1_records):
        if entity_ids is not None and rec.entity_id not in entity_ids:
            continue
        matches = truth.get(rec.entity_id)
        if not matches:
            continue
        got = cands_idx.get(i, set())
        for mid in matches:
            total += 1
            j = other_by_id.get(mid)
            if j is not None and j in got:
                found += 1
    return (found / total if total else 1.0), found, total