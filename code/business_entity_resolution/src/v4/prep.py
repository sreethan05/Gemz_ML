"""v4 prep: build train-pool indexes, mine training pairs, save eval data.

Production-honest design:
- Pools are the FULL per-country S2/S3 train pools (US ~6.2M, India ~4.1M) —
  the same scale the model faces at test time.
- Eval S1 entities are a deterministic md5 holdout (~4%) — never seen in training.
- Positives are NOT guaranteed index placement: they must survive the same
  blocking as at test time.
"""
import gc
import hashlib
import pickle
import sys
import time
from array import array
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import (PackedPool, CSRIndex, EntityRecord, ensure_hashseed,
                     keys_of, key_hashes, compute_pair_features, query_cascade,
                     N_FEATURES, build_clusters, expand_with_siblings)

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "scratch" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

N_TRAIN_PER_CTRY = 100_000
N_EVAL_PER_CTRY = 25_000
QUERY_TRAIN = 20   # candidates mined for hard negatives
QUERY_EVAL = 40    # candidates saved for calibration
NEG_PER_ENTITY = 10
# Match run_test.py's production index layout.  query_cascade asks each CSR
# chunk for a share of its intermediate top-k, so changing the number of
# chunks changes the candidates and ranking even when the underlying records
# are identical.  The old 1.2M chunking made local calibration over-optimistic
# relative to the single-CSR production inference path.
CSR_CHUNK = 10_000_000
BUILD_CACHE = 100_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_of(eid: str) -> int:
    """0 = train, 1 = eval (deterministic)."""
    h = int(hashlib.md5(eid.encode()).hexdigest()[:8], 16) % 1000
    return 1 if h >= 960 else 0


def build_pool(country: str) -> PackedPool:
    pool = PackedPool()
    pool.cache_cap = 1  # disable cache during build (every record seen once)
    t0 = time.time()
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4 and p[3].strip().lower() == country:
                    pool.add_raw(p[0], p[1], p[2])
                    if pool.n % 1_000_000 == 0:
                        log(f"[{country}] added {pool.n:,} records "
                            f"({time.time()-t0:.0f}s)")
    log(f"[{country}] pool raw add done: {pool.n:,} records in {time.time()-t0:.0f}s")
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = BUILD_CACHE
    log(f"[{country}] pool finalized")
    return pool


def build_csrs(pool: PackedPool):
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        t0 = time.time()
        j1 = min(j0 + CSR_CHUNK, pool.n)
        csrs.append(CSRIndex.build(pool, j0, j1))
        log(f"  CSR chunk [{j0:,}:{j1:,}] built in {time.time()-t0:.0f}s "
            f"({len(csrs[-1].J):,} postings)")
        pool._cache.clear()
        gc.collect()
    return csrs


def query_top(pool: PackedPool, csrs, rec: EntityRecord, topk: int):
    js, _ = query_cascade(pool, csrs, rec, topk)
    return [int(j) for j in js]

def select_s1(country: str):
    train_recs, eval_recs = [], []
    with open(TRAIN / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3].strip().lower() == country:
                if split_of(p[0]) == 1:
                    if len(eval_recs) < N_EVAL_PER_CTRY:
                        eval_recs.append(EntityRecord(p[0], p[1], p[2]))
                else:
                    if len(train_recs) < N_TRAIN_PER_CTRY:
                        train_recs.append(EntityRecord(p[0], p[1], p[2]))
            if len(train_recs) >= N_TRAIN_PER_CTRY and len(eval_recs) >= N_EVAL_PER_CTRY:
                # keep scanning only if either bucket for this country unfilled
                pass
    return train_recs, eval_recs


def load_truth_subset(needed: set) -> dict:
    truth = {}
    with open(TRAIN / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed:
                truth[p[0]] = set(x for x in p[1].split(",") if x) if len(p) > 1 and p[1] else set()
    return truth


def process_country(country: str, eval_only: bool = False, expand: bool = False):
    log(f"========== {country.upper()} ==========")
    pool = build_pool(country)
    csrs = build_csrs(pool)
    cluster_maps = build_clusters(pool) if expand else None
    if expand:
        gc.collect()
        log(f"[{country}] sibling clusters built")
    log(f"[{country}] index ready; querying S1 selection...")

    t0 = time.time()
    train_recs, eval_recs = select_s1(country)
    log(f"[{country}] selected {len(train_recs):,} train / {len(eval_recs):,} eval S1 "
        f"in {time.time()-t0:.0f}s")

    needed = {r.id for r in train_recs} | {r.id for r in eval_recs}
    truth = load_truth_subset(needed)
    log(f"[{country}] truth loaded for {len(truth):,} S1 ids")

    if eval_only:
        log(f"[{country}] eval-only mode: skipping mining")
    # ---- mine training pairs -------------------------------------------------
    X_flat = array("f")
    y_rows = array("b")
    n_pos = n_neg = 0
    t0 = time.time()
    for i, r in enumerate(train_recs if not eval_only else []):
        true_set = truth.get(r.id, set())
        pos_js = []
        for m in true_set:
            j = pool.lookup_id(m)
            if j >= 0:
                pos_js.append(j)
        cand_js = query_top(pool, csrs, r, QUERY_TRAIN)
        neg_js = [j for j in cand_js if pool.get_id(j) not in true_set][:NEG_PER_ENTITY]
        for j in pos_js:
            X_flat.extend(compute_pair_features(r, pool.record(j)))
            y_rows.append(1)
        for j in neg_js:
            X_flat.extend(compute_pair_features(r, pool.record(j)))
            y_rows.append(0)
        n_pos += len(pos_js)
        n_neg += len(neg_js)
        if (i + 1) % 10_000 == 0:
            log(f"[{country}] mining {i+1:,}/{len(train_recs):,} entities "
                f"({n_pos:,} pos / {n_neg:,} neg, {time.time()-t0:.0f}s)")
    n_pairs = len(y_rows)
    X = np.frombuffer(X_flat, dtype=np.float32).reshape(n_pairs, N_FEATURES).copy()
    y = np.frombuffer(y_rows, dtype=np.int8).copy()
    del X_flat, y_rows
    gc.collect()
    log(f"[{country}] mined {len(X):,} pairs ({n_pos:,} pos / {n_neg:,} neg) "
        f"in {time.time()-t0:.0f}s -> {country}_train_X.npy")
    np.save(OUT / f"{country}_train_X.npy", X)
    np.save(OUT / f"{country}_train_y.npy", y)
    del X, y
    gc.collect()

    # ---- eval candidates -----------------------------------------------------
    feats_flat = array("f")
    s1_idx = array("i")
    cand_j = array("i")
    cand_ids = []
    t0 = time.time()
    for i, r in enumerate(eval_recs):
        cand_js = query_top(pool, csrs, r, QUERY_EVAL)
        if cluster_maps is not None and cand_js:
            cand_js = expand_with_siblings(pool, cluster_maps, cand_js, extra=15).tolist()
        cand_ids.append([pool.get_id(j) for j in cand_js])
        for j in cand_js:
            feats_flat.extend(compute_pair_features(r, pool.record(j)))
            s1_idx.append(i)
            cand_j.append(j)
    n_ep = len(s1_idx)
    feats = np.frombuffer(feats_flat, dtype=np.float32).reshape(n_ep, N_FEATURES).copy()
    del feats_flat
    gc.collect()
    log(f"[{country}] eval pairs: {len(feats):,} in {time.time()-t0:.0f}s")
    np.save(OUT / f"{country}_eval_feats.npy", feats)
    np.save(OUT / f"{country}_eval_s1idx.npy", np.asarray(s1_idx, dtype=np.int32))
    np.save(OUT / f"{country}_eval_j.npy", np.asarray(cand_j, dtype=np.int32))
    with open(OUT / f"{country}_eval_candids.pkl", "wb") as f:
        pickle.dump(cand_ids, f, protocol=4)
    with open(OUT / f"{country}_eval_meta.pkl", "wb") as f:
        pickle.dump({
            "eval_recs": [(r.id, r.name_norm, r.addr_norm) for r in eval_recs],
            "eval_truth": {r.id: sorted(truth.get(r.id, set())) for r in eval_recs},
        }, f, protocol=4)
    log(f"[{country}] eval artifacts saved")

    # ---- blocking recall at caps (the recall ceiling) ------------------------
    tot_tp = 0
    rec_at = {c: 0 for c in (10, 12, 15, 20, 25, 30, 40)}
    n_truth_pairs = 0
    for i, r in enumerate(eval_recs):
        tset_ = truth.get(r.id, set())
        if not tset_:
            continue
        ids = cand_ids[i]
        n_truth_pairs += len(tset_)
        found = 0
        for k, cid in enumerate(ids):
            if cid in tset_:
                found += 1
                for c in rec_at:
                    if k < c:
                        rec_at[c] += 1
        tot_tp += found
    log(f"[{country}] BLOCKING RECALL: total {tot_tp:,}/{n_truth_pairs:,} "
        f"({tot_tp/max(n_truth_pairs,1):.4f})")
    for c, v in rec_at.items():
        log(f"    cap={c:2d}: {v/max(n_truth_pairs,1):.4f}")

    del pool, csrs
    gc.collect()


def main():
    ensure_hashseed()
    eval_only = "--eval-only" in sys.argv
    expand = "--expand" in sys.argv
    for country in ("us", "india"):
        if (OUT / f"{country}_eval_meta.pkl").exists():
            log(f"[{country}] artifacts already exist - skipping (resume mode)")
            continue
        process_country(country, eval_only=eval_only, expand=expand)
    log("PREP COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("FATAL: out of memory. Close other applications and re-run.", flush=True)
        sys.exit(3)
