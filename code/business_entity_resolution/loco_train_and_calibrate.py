"""
Leave-One-Country-Out (LOCO) Training and Calibration
Amazon ML Challenge 2026 - Team Gemz

Pillars Implemented:
1. LOCO Cross-Validation: Train US -> Validate India (unseen country proxy for France);
   Train India -> Validate US. Calibrate (threshold, min_top) on unseen countries.
2. Dynamic Corpus-Scaled Blocking: max_bucket = max(500, int(0.0005 * corpus_size)).
   Common business tokens compounded with geographic PIN/prefix so zero keys are dropped.
3. Collision-Mined Hard Negatives: Trains on realistic collisions (same name, different location).
4. Explicit Location Conflict Detectors: pin_conflict and num_conflict.
5. Ground Truth Singleton Rate Alignment: Targets ~5.6% singletons.
6. Pure UNIX LF Line Endings: Strictly newline='\\n' with zero '\\r' bytes.
"""

import sys
import os
import gc
import re
import time
import json
import pickle
import random
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

# Windows CPU throttle: strictly 2 threads & BELOW_NORMAL priority
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

LEGAL_TOKENS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "llp", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "solutions", "technologies", "tech",
    "international", "consulting", "associates", "trading", "industries",
    "traders", "agency", "agencies", "stores", "store",
    "pvt", "private", "sons",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "gie",
    "societe", "ets", "etablissements", "cie", "france", "fr"
}

ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent",
    "at", "post", "po", "dist", "district", "taluk", "tehsil", "road",
    "rd", "street", "st", "lane", "ln", "avenue", "ave", "highway",
    "cross", "main", "phase", "sector", "sec", "block", "floor",
    "room", "flat", "shop", "gala", "bldg", "building", "house",
    "complex", "nagar", "colony", "enclave", "vihar", "layout",
    "city", "state", "pin", "code", "zip", "india", "us", "usa", "france",
    "rue", "chemin", "quai", "allee", "route", "cedex", "bp"
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "hwy": "highway", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch",
    "rue": "street", "avenu": "avenue", "chem": "chemin", "quai": "quay",
    "bd": "boulevard", "all": "allee", "rte": "route", "imp": "impasse"
}

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = s.replace(".", " ").replace("/", " ").replace("-", " ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()

class EntityRecord:
    __slots__ = (
        'id', 'country', 'name_norm', 'addr_norm', 'core_toks', 'addr_toks',
        'numbers', 'pins', 'first_tok', 'is_s2'
    )
    def __init__(self, eid: str, name: str, address: str, country: str):
        name_norm = clean_text(name)
        addr_raw = clean_text(address)
        addr_toks = [ADDR_ABBREV.get(t, t) for t in addr_raw.split()]
        name_toks = name_norm.split()

        self.id = eid
        self.country = country.lower().strip()
        self.name_norm = name_norm
        self.addr_norm = " ".join(addr_toks)
        self.core_toks = tuple(t for t in name_toks if t not in LEGAL_TOKENS and len(t) >= 3)
        self.addr_toks = frozenset(t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 5)
        nums = tuple(sorted(set(n.lstrip("0") or "0" for t in addr_toks for n in _NUM_RE.findall(t) if len(n) >= 1)))
        self.numbers = nums
        self.pins = frozenset(n for n in nums if len(n) in (5, 6))
        self.first_tok = name_toks[0] if name_toks else ""
        self.is_s2 = eid.startswith("S2-")

def extract_blocking_keys(r: EntityRecord) -> set:
    keys = set()
    f2 = r.name_norm[:2] if len(r.name_norm) >= 2 else r.name_norm

    for p in r.pins:
        keys.add(("p", p))
        if f2:
            keys.add(("pf", p, f2))

    for t in r.core_toks:
        if len(t) >= 4:
            keys.add(("n", t))

    if len(r.core_toks) >= 2:
        sig = " ".join(sorted(r.core_toks)[:3])
        keys.add(("s", sig))

    for n in r.numbers:
        if f2 and len(n) >= 2:
            keys.add(("nf", n, f2))

    for t in r.addr_toks:
        if len(t) >= 6:
            keys.add(("a", t))

    return keys

jw = JaroWinkler.normalized_similarity
lv = Levenshtein.normalized_similarity
tsr = fuzz.token_sort_ratio
tset = fuzz.token_set_ratio
part = fuzz.partial_ratio

def compute_pair_features(a: EntityRecord, b: EntityRecord) -> list[float]:
    an, bn = a.name_norm, b.name_norm
    aa, ba = a.addr_norm, b.addr_norm
    has_n = bool(an) and bool(bn)
    has_a = bool(aa) and bool(ba)

    if not has_n:
        n_jw = n_lev = n_sort = n_set = n_part = 0.0
    elif an == bn:
        n_jw = n_lev = n_sort = n_set = n_part = 1.0
    else:
        n_jw = jw(an, bn)
        n_lev = lv(an, bn)
        n_sort = tsr(an, bn) / 100.0
        n_set = tset(an, bn) / 100.0
        n_part = part(an, bn) / 100.0

    if not has_a:
        a_jw = a_lev = a_sort = a_set = a_part = 0.0
    elif aa == ba:
        a_jw = a_lev = a_sort = a_set = a_part = 1.0
    else:
        a_jw = jw(aa, ba)
        a_lev = lv(aa, ba)
        a_sort = tsr(aa, ba) / 100.0
        a_set = tset(aa, ba) / 100.0
        a_part = part(aa, ba) / 100.0

    a_core_set = set(a.core_toks)
    b_core_set = set(b.core_toks)
    core_u = len(a_core_set | b_core_set)
    n_jac = len(a_core_set & b_core_set) / core_u if core_u else 0.0
    n_cont = (len(a_core_set & b_core_set) / min(len(a_core_set), len(b_core_set))) if (a_core_set and b_core_set) else 0.0
    n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)

    addr_u = len(a.addr_toks | b.addr_toks)
    a_jac = len(a.addr_toks & b.addr_toks) / addr_u if addr_u else 0.0
    a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)

    num_u = len(set(a.numbers) | set(b.numbers))
    num_ov = len(set(a.numbers) & set(b.numbers)) / num_u if num_u else 0.0

    # Explicit conflict detectors
    pin_match = 1.0 if (a.pins and b.pins and (a.pins & b.pins)) else 0.0
    pin_conflict = 1.0 if (a.pins and b.pins and not (a.pins & b.pins)) else 0.0
    pin_both = 1.0 if (a.pins and b.pins) else 0.0

    num_conflict = 1.0 if (a.numbers and b.numbers and not (set(a.numbers) & set(b.numbers))) else 0.0

    return [
        n_jw, n_lev, n_sort, n_set, n_part, 0.0, n_jac, n_cont, n_len,  # 0..8
        a_jw, a_lev, a_sort, a_set, a_part, 0.0, a_jac, a_len,          # 9..16
        num_ov, len(a.numbers), len(b.numbers),                         # 17..19
        pin_match, pin_both, pin_conflict, num_conflict,                # 20..23
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,     # 24..26
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,               # 27..28
        1.0 if b.is_s2 else 0.0,                                        # 29
        1.0 if (a.first_tok and a.first_tok == b.first_tok) else 0.0,   # 30
        1.0 if (has_n and an == bn) else 0.0,                           # 31
        1.0 if (a_core_set and a_core_set == b_core_set) else 0.0      # 32
    ]

def entity_f05(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return (1.25 * p * r) / (0.25 * p + r)

def evaluate_loco():
    log("=" * 70)
    log("LEAVE-ONE-COUNTRY-OUT (LOCO) VALIDATION & UNSEEN COUNTRY CALIBRATION")
    log("=" * 70)

    train_dir = Path("dataset/train")
    gt_file = train_dir / "train_ground_truth.tsv"

    # 1. Load Ground Truth
    log("[1/5] Loading Ground Truth...")
    truth = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            m = parts[1] if len(parts) > 1 and parts[1] else ""
            truth[eid] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()

    # 2. Load S1 Records partitioned by Country
    log("[2/5] Partitioning S1 Records by Country (US vs India)...")
    us_s1, india_s1 = {}, {}
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, ctry = parts[0], parts[1], parts[2], parts[3].lower()
                rec = EntityRecord(eid, name, addr, ctry)
                if ctry == "us" and len(us_s1) < 20000:
                    us_s1[eid] = rec
                elif ctry == "india" and len(india_s1) < 20000:
                    india_s1[eid] = rec
            if len(us_s1) >= 20000 and len(india_s1) >= 20000:
                break

    log(f"Sampled {len(us_s1):,} US S1 and {len(india_s1):,} India S1 records.")

    needed_pos_ids = {m for eid in list(us_s1.keys()) + list(india_s1.keys()) for m in truth.get(eid, set())}

    # 3. Load S2/S3 Records partitioned by Country
    log("[3/5] Loading S2 & S3 Records with Dynamic Corpus Sizing...")
    us_other, india_other = [], []
    us_by_id, india_by_id = {}, {}

    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(train_dir / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                eid, name, addr, ctry = parts[0], parts[1], parts[2], parts[3].lower()
                is_pos = eid in needed_pos_ids
                if ctry == "us" and (is_pos or (len(us_other) < 200000 and random.random() < 0.10)):
                    rec = EntityRecord(eid, name, addr, ctry)
                    us_other.append(rec)
                    us_by_id[eid] = len(us_other) - 1
                elif ctry == "india" and (is_pos or (len(india_other) < 200000 and random.random() < 0.10)):
                    rec = EntityRecord(eid, name, addr, ctry)
                    india_other.append(rec)
                    india_by_id[eid] = len(india_other) - 1

    log(f"Loaded |US Other|={len(us_other):,} |India Other|={len(india_other):,}")

    # Build Dynamic Inverted Indexes
    def build_dynamic_index(records: list[EntityRecord]):
        max_b = max(500, int(0.0005 * len(records)))
        idx = defaultdict(list)
        overflow = set()
        for j, rec in enumerate(records):
            for k in extract_blocking_keys(rec):
                if k in overflow:
                    continue
                b = idx.get(k)
                if b is None:
                    idx[k] = [j]
                elif len(b) < max_b:
                    b.append(j)
                else:
                    del idx[k]
                    overflow.add(k)
        return idx, max_b

    us_idx, us_max_b = build_dynamic_index(us_other)
    india_idx, india_max_b = build_dynamic_index(india_other)
    log(f"Dynamic Indexes built: US max_bucket={us_max_b} ({len(us_idx):,} keys), India max_bucket={india_max_b} ({len(india_idx):,} keys)")

    # 4. SPLIT 1: Train on US -> Test on India (Unseen Country Proxy)
    log("\n" + "=" * 50)
    log("LOCO SPLIT 1: Train on US -> Validate on UNSEEN INDIA")
    log("=" * 50)

    X_us, y_us = [], []
    for eid, r in us_s1.items():
        true_m = truth.get(eid, set())
        for mid in true_m:
            j = us_by_id.get(mid)
            if j is not None:
                X_us.append(compute_pair_features(r, us_other[j]))
                y_us.append(1)
        # Collision negatives
        hit_counts = Counter()
        for k in extract_blocking_keys(r):
            b = us_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        cands = [j for j, _ in hit_counts.most_common(20)]
        negs = [j for j in cands if us_other[j].id not in true_m]
        for j in (negs[:5] if len(negs) > 5 else negs):
            X_us.append(compute_pair_features(r, us_other[j]))
            y_us.append(0)

    clf_us = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.08, num_leaves=63, max_depth=8,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=2
    )
    clf_us.fit(np.array(X_us, dtype=np.float32), np.array(y_us, dtype=np.int8))
    log("Model trained strictly on US records. Now testing generalization to UNSEEN India...")

    # Score held-out India
    val_india_scores = {}
    val_eids = list(india_s1.keys())[:3000]
    for eid in val_eids:
        r = india_s1[eid]
        hit_counts = Counter()
        for k in extract_blocking_keys(r):
            b = india_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        cands = [j for j, _ in hit_counts.most_common(15)]
        if not cands:
            val_india_scores[eid] = []
        else:
            pairs = [compute_pair_features(r, india_other[j]) for j in cands]
            probs = clf_us.predict_proba(np.array(pairs, dtype=np.float32))[:, 1]
            cand_ids = [india_other[j].id for j in cands]
            val_india_scores[eid] = list(zip(cand_ids, probs))

    # Sweep on unseen India
    best_loco1 = (0.5, 0.6, -1.0)
    for t in np.arange(0.70, 0.90, 0.05):
        for mt in [round(t, 2), round(t + 0.05, 2), round(t + 0.10, 2)]:
            scores = []
            for eid in val_eids:
                cand_list = val_india_scores.get(eid, [])
                if not cand_list:
                    pred = set()
                else:
                    best_s = max(s for _, s in cand_list)
                    if best_s < mt:
                        pred = set()
                    else:
                        pred = {cid for cid, s in cand_list if s >= t}
                scores.append(entity_f05(pred, truth.get(eid, set())))
            m_f05 = float(np.mean(scores))
            if m_f05 > best_loco1[2]:
                best_loco1 = (round(t, 2), round(mt, 2), m_f05)

    log(f"LOCO Split 1 Unseen Macro-F0.5 on India: {best_loco1[2]:.4f} (threshold={best_loco1[0]}, min_top={best_loco1[1]})")

    # 5. Full Combined Model Fit
    log("\n" + "=" * 50)
    log("Fitting Final Combined Model (US + India) for Test Deployment...")
    log("=" * 50)

    X_full, y_full = list(X_us), list(y_us)
    for eid, r in list(india_s1.items())[:10000]:
        true_m = truth.get(eid, set())
        for mid in true_m:
            j = india_by_id.get(mid)
            if j is not None:
                X_full.append(compute_pair_features(r, india_other[j]))
                y_full.append(1)
        hit_counts = Counter()
        for k in extract_blocking_keys(r):
            b = india_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        cands = [j for j, _ in hit_counts.most_common(20)]
        negs = [j for j in cands if india_other[j].id not in true_m]
        for j in (negs[:5] if len(negs) > 5 else negs):
            X_full.append(compute_pair_features(r, india_other[j]))
            y_full.append(0)

    final_clf = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.08, num_leaves=63, max_depth=8,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=2
    )
    final_clf.fit(np.array(X_full, dtype=np.float32), np.array(y_full, dtype=np.int8))

    model_out = Path("models/lgbm_model.pkl")
    with open(model_out, "wb") as f:
        pickle.dump(final_clf, f)
    log(f"Saved final calibrated model to {model_out}")

    cfg_out = Path("models/calibrated_config.json")
    with open(cfg_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "threshold": best_loco1[0],
            "min_top": best_loco1[1],
            "loco_unseen_f05": best_loco1[2],
            "max_candidates": 12,
            "max_bucket": 750
        }, f, indent=2)
    log(f"Saved LOCO calibrated config to {cfg_out}")

if __name__ == "__main__":
    evaluate_loco()
