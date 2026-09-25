"""
Train and Save LightGBM Model for Business Entity Resolution
Team Gemz - Amazon ML Challenge 2026
"""

import sys
import gc
import csv
import re
import json
import time
import random
import pickle
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

LEGAL_TOKENS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "pvt", "private", "llc", "llp", "gmbh", "sa",
    "sarl", "bv", "nv", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "solutions", "technologies", "tech",
    "international", "consulting", "associates", "sons", "trading"
}

ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent",
    "at", "post", "po", "dist", "district", "taluk", "tehsil", "road",
    "rd", "street", "st", "lane", "ln", "avenue", "ave", "highway",
    "cross", "main", "phase", "sector", "sec", "block", "floor",
    "room", "flat", "shop", "gala", "bldg", "building", "house",
    "complex", "nagar", "colony", "enclave", "vihar", "layout",
    "city", "state", "pin", "code", "zip", "india", "us", "usa", "france"
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "hwy": "highway", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch",
    "rue": "street", "avenu": "avenue", "chem": "chemin", "quai": "quay"
}

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = s.replace(".", " ").replace("/", " ").replace("-", " ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def build_record(entity_id: str, name: str, address: str, country: str) -> dict:
    name_norm = clean_text(name)
    addr_toks = [ADDR_ABBREV.get(t, t) for t in clean_text(address).split()]
    addr_norm = " ".join(addr_toks)
    name_toks = name_norm.split()

    core_toks = [t for t in name_toks if t not in LEGAL_TOKENS and len(t) >= 2]
    addr_clean = [t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 5]
    nums = tuple(sorted(set(n.lstrip("0") or "0" for t in addr_toks for n in _NUM_RE.findall(t) if len(n) >= 1)))
    pins = frozenset(n for n in nums if len(n) in (5, 6))

    f2 = name_norm[:2] if len(name_norm) >= 2 else name_norm
    first_tok = name_toks[0] if name_toks else ""

    return {
        "id": entity_id, "name_norm": name_norm, "addr_norm": addr_norm, "ctry": clean_text(country) or "unknown",
        "core_toks": tuple(core_toks), "addr_toks": frozenset(addr_clean), "numbers": nums, "pins": pins,
        "f2": f2, "first_tok": first_tok
    }


def extract_blocking_keys(r: dict) -> set:
    keys = set()
    c = r["ctry"]
    f2 = r["f2"]
    ctoks = r["core_toks"]

    for t in ctoks:
        if len(t) >= 3:
            keys.add(("nt", c, t))

    if ctoks:
        sig = " ".join(sorted(ctoks)[:3])
        keys.add(("sig", c, sig))

    for t in r["addr_toks"]:
        keys.add(("at", c, t))

    for p in r["pins"]:
        keys.add(("pin", c, p))
        if f2:
            keys.add(("pin_f2", c, p, f2))

    for n in r["numbers"]:
        if f2 and len(n) >= 2:
            keys.add(("num_f2", c, n, f2))

    return keys


def build_blocking_index(other_recs: list[dict], max_bucket: int = 250):
    df_counts = Counter()
    for r in other_recs:
        for k in extract_blocking_keys(r):
            df_counts[k] += 1

    index = defaultdict(list)
    for j, r in enumerate(other_recs):
        for k in extract_blocking_keys(r):
            if df_counts[k] <= max_bucket:
                index[k].append(j)

    return index


def query_candidates(r: dict, index: dict, max_candidates: int = 40) -> list[int]:
    hit_counts = Counter()
    for k in extract_blocking_keys(r):
        if k in index:
            for j in index[k]:
                hit_counts[j] += 1
    return [j for j, _ in hit_counts.most_common(max_candidates)]


def compute_pair_features(a: dict, b: dict) -> list[float]:
    an, bn = a["name_norm"], b["name_norm"]
    aa, ba = a["addr_norm"], b["addr_norm"]
    has_n = bool(an) and bool(bn)
    has_a = bool(aa) and bool(ba)

    jw = JaroWinkler.normalized_similarity
    lv = Levenshtein.normalized_similarity
    tsr = fuzz.token_sort_ratio
    tset = fuzz.token_set_ratio
    part = fuzz.partial_ratio

    n_jw = jw(an, bn) if has_n else 0.0
    n_lev = lv(an, bn) if has_n else 0.0
    n_sort = tsr(an, bn) / 100.0 if has_n else 0.0
    n_set = tset(an, bn) / 100.0 if has_n else 0.0
    n_part = part(an, bn) / 100.0 if has_n else 0.0

    a_jw = jw(aa, ba) if has_a else 0.0
    a_lev = lv(aa, ba) if has_a else 0.0
    a_sort = tsr(aa, ba) / 100.0 if has_a else 0.0
    a_set = tset(aa, ba) / 100.0 if has_a else 0.0
    a_part = part(aa, ba) / 100.0 if has_a else 0.0

    a_core_set = set(a["core_toks"])
    b_core_set = set(b["core_toks"])
    core_u = len(a_core_set | b_core_set)
    n_jac = len(a_core_set & b_core_set) / core_u if core_u else 0.0
    n_cont = (len(a_core_set & b_core_set) / min(len(a_core_set), len(b_core_set))) if (a_core_set and b_core_set) else 0.0
    n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)

    addr_u = len(a["addr_toks"] | b["addr_toks"])
    a_jac = len(a["addr_toks"] & b["addr_toks"]) / addr_u if addr_u else 0.0
    a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)

    num_u = len(set(a["numbers"]) | set(b["numbers"]))
    num_ov = len(set(a["numbers"]) & set(b["numbers"])) / num_u if num_u else 0.0

    return [
        n_jw, n_lev, n_sort, n_set, n_part, 0.0, n_jac, n_cont, n_len,
        a_jw, a_lev, a_sort, a_set, a_part, 0.0, a_jac, a_len,
        num_ov, len(a["numbers"]), len(b["numbers"]),
        1.0 if (a["pins"] and a["pins"] & b["pins"]) else 0.0,
        1.0 if (a["pins"] and b["pins"]) else 0.0,
        1.0 if a["ctry"] == b["ctry"] else 0.0,
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,
        1.0 if b["id"].startswith("S2-") else 0.0,
        1.0 if (a["first_tok"] and a["first_tok"] == b["first_tok"]) else 0.0,
        1.0 if (has_n and an == bn) else 0.0,
        1.0 if (a_core_set and a_core_set == b_core_set) else 0.0
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


def sweep_thresholds(val_scores: dict, truth: dict):
    best = (0.50, 0.60, -1.0)
    entities = list(truth.keys())
    for t in np.arange(0.30, 0.85, 0.05):
        for mt in [round(float(t), 2), round(float(t + 0.05), 2), round(float(t + 0.10), 2), 0.90, 0.95]:
            scores = []
            for e in entities:
                cand_list = val_scores.get(e, [])
                if not cand_list:
                    pred = set()
                else:
                    best_score = max(s for _, s in cand_list)
                    if best_score < mt:
                        pred = set()
                    else:
                        pred = {cid for cid, s in cand_list if s >= t}
                scores.append(entity_f05(pred, truth[e]))
            mean_f = float(np.mean(scores))
            if mean_f > best[2]:
                best = (round(float(t), 2), round(float(mt), 2), mean_f)
    return best


def load_filtered(path: Path, target_ids: set = None, sample_prob: float = 1.0, max_extra: int = 0) -> list[dict]:
    recs = []
    extra_count = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=150000, quoting=csv.QUOTE_NONE):
        for r in chunk.itertuples(index=False):
            eid = str(r.entity_id).strip()
            is_target = target_ids is not None and eid in target_ids
            if is_target:
                recs.append(build_record(eid, r.business_name, r.business_address, r.country))
            elif sample_prob < 1.0 and extra_count < max_extra:
                if random.random() < sample_prob:
                    recs.append(build_record(eid, r.business_name, r.business_address, r.country))
                    extra_count += 1
            elif target_ids is None:
                recs.append(build_record(eid, r.business_name, r.business_address, r.country))
    return recs


def main():
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "lgbm_model.pkl"
    thresh_file = models_dir / "thresholds.json"

    train_dir = Path("dataset/train")

    log("=" * 60)
    log("TRAINING & PERSISTING LIGHTGBM MODEL — TEAM GEMZ")
    log("=" * 60)

    # 1. Ground Truth
    log("[1/4] Loading Ground Truth labels...")
    gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    truth = {}
    for r in gt_df.itertuples(index=False):
        raw = str(getattr(r, "matched_entity_ids", "") or "").strip()
        truth[r.source1_entity_id] = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()

    random.seed(42)
    np.random.seed(42)

    s1_all_ids = list(truth.keys())
    random.shuffle(s1_all_ids)
    train_sample_size = 50000
    sampled_s1_ids = set(s1_all_ids[:train_sample_size])

    val_size = int(train_sample_size * 0.15)
    val_ids = set(list(sampled_s1_ids)[:val_size])
    fit_ids = sampled_s1_ids - val_ids
    log(f"split: {len(fit_ids):,} fit / {len(val_ids):,} validation S1 entities")

    needed_pos_ids = {m for eid in sampled_s1_ids for m in truth.get(eid, set())}
    log(f"identified {len(needed_pos_ids):,} true match targets needed for training slice")

    # 2. Training Source Records
    log("[2/4] Loading Training Source Records...")
    s1_train = load_filtered(train_dir / "train_source1.tsv", target_ids=sampled_s1_ids)
    s2_train = load_filtered(train_dir / "train_source2.tsv", target_ids=needed_pos_ids, sample_prob=0.03, max_extra=100000)
    s3_train = load_filtered(train_dir / "train_source3.tsv", target_ids=needed_pos_ids, sample_prob=0.03, max_extra=100000)
    other_train = s2_train + s3_train
    other_by_id = {r["id"]: j for j, r in enumerate(other_train)}
    log(f"train slice: |S1|={len(s1_train):,} |S2+S3|={len(other_train):,}")

    # 3. Blocking
    log("[3/4] Blocking & Featurizing Candidate Pairs...")
    index = build_blocking_index(other_train, max_bucket=250)
    log(f"  blocking index built with {len(index):,} active buckets")

    cands_idx = {}
    for i, r in enumerate(s1_train):
        cands_idx[i] = query_candidates(r, index, max_candidates=40)

    # 4. LightGBM Training & Sweep
    X_train, y_train = [], []
    val_scores = {e: [] for e in val_ids}

    for i, r in enumerate(s1_train):
        is_val = r["id"] in val_ids
        true_m = truth.get(r["id"], set())
        got = cands_idx[i]
        got_set = set(got)

        if not is_val:
            negs = [j for j in got if other_train[j]["id"] not in true_m]
            if len(negs) > 10:
                negs = random.sample(negs, 10)
            pos_in_got = [j for j in got if other_train[j]["id"] in true_m]

            for j in negs:
                X_train.append(compute_pair_features(r, other_train[j]))
                y_train.append(0)
            for j in pos_in_got:
                X_train.append(compute_pair_features(r, other_train[j]))
                y_train.append(1)
            for m in true_m:
                j = other_by_id.get(m)
                if j is not None and j not in got_set:
                    X_train.append(compute_pair_features(r, other_train[j]))
                    y_train.append(1)
        else:
            for j in got:
                val_scores[r["id"]].append((other_train[j]["id"], compute_pair_features(r, other_train[j])))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    log(f"training set: {len(X_train):,} pairs ({y_train.sum():,} positive, {len(y_train) - y_train.sum():,} negative)")

    pos_w = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)
    clf = lgb.LGBMClassifier(
        n_estimators=450, learning_rate=0.06, num_leaves=63,
        scale_pos_weight=min(pos_w, 15.0), subsample=0.9,
        random_state=42, n_jobs=-1, verbose=-1
    )
    log("Fitting LightGBM model...")
    clf.fit(X_train, y_train)
    del X_train, y_train, s1_train, s2_train, s3_train, other_train, index; gc.collect()

    log("[4/4] Tuning decision rule on validation split...")
    val_probs = {}
    for e, pair_list in val_scores.items():
        if not pair_list:
            val_probs[e] = []
        else:
            X_v = np.array([f for _, f in pair_list], dtype=np.float32)
            p = clf.predict_proba(X_v)[:, 1]
            val_probs[e] = [(oid, float(score)) for (oid, _), score in zip(pair_list, p)]

    val_truth = {e: truth.get(e, set()) for e in val_ids}
    best_t, best_mt, val_f05 = sweep_thresholds(val_probs, val_truth)
    log(f"Tuned decision rule: threshold={best_t:.2f} min_top={best_mt:.2f} -> validation macro F_0.5 = {val_f05:.4f}")

    # Save to disk
    with open(model_file, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(thresh_file, "w") as f:
        json.dump({"threshold": best_t, "min_top": best_mt, "val_f05": val_f05}, f, indent=2)

    log(f"SAVED: Model saved to {model_file.resolve()} and {thresh_file.resolve()}")


if __name__ == "__main__":
    main()
