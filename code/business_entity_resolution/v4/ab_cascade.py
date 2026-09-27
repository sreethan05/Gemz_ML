"""A/B: does the query_cascade improve blocking recall over plain-IDF ranking?

Rebuilds the US pool/CSRs, runs the cascade on a sample of the SAME eval
entities used in prep, and compares hit@cap against the saved candidates.
Also measures per-entity query time to budget the full test run.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import (PackedPool, CSRIndex, EntityRecord, query_cascade,
                     keys_of, key_hashes)

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "scratch" / "v4"
CSR_CHUNK = 1_600_000


def main(n_ent_sample: int = 5000):
    with open(OUT / "us_eval_candids.pkl", "rb") as f:
        cand_ids = pickle.load(f)
    with open(OUT / "us_eval_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    eval_recs = [(r[0], EntityRecord(r[0], r[1], r[2])) for r in meta["eval_recs"]]
    truth = {k: set(v) for k, v in meta["eval_truth"].items()}
    eval_recs = eval_recs[:n_ent_sample]

    # rebuild US pool + CSRs
    t0 = time.time()
    pool = PackedPool()
    pool.cache_cap = 1
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4 and p[3].strip().lower() == "us":
                    pool.add_raw(p[0], p[1], p[2])
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = 200_000
    print(f"pool rebuilt: {pool.n:,} records in {time.time()-t0:.0f}s")
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        csrs.append(CSRIndex.build(pool, j0, min(j0 + CSR_CHUNK, pool.n)))
        pool._cache.clear()
    print("csrs ready")

    caps = (10, 12, 15, 20, 25, 30, 40)
    base_hit = {c: 0 for c in caps}
    casc_hit = {c: 0 for c in caps}
    tot_pairs = 0
    t0 = time.time()
    for i, (eid, rec) in enumerate(eval_recs):
        tr = truth.get(eid, set())
        tot_pairs += len(tr)
        base = cand_ids[i]
        for c in caps:
            base_hit[c] += sum(1 for cid in base[:c] if cid in tr)
        js, _ = query_cascade(pool, csrs, rec, 40)
        casc_ids = [pool.get_id(int(j)) for j in js]
        for c in caps:
            casc_hit[c] += sum(1 for cid in casc_ids[:c] if cid in tr)
        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"  {i+1:,} entities, {el:.0f}s ({el*1000/(i+1):.2f} ms/entity)")
    print(f"\ntotal true pairs: {tot_pairs:,}")
    print(f"{'cap':>4} | {'plain-IDF':>9} | {'cascade':>9} | gain")
    for c in caps:
        b = base_hit[c] / tot_pairs
        k = casc_hit[c] / tot_pairs
        print(f"{c:>4} | {b:>9.4f} | {k:>9.4f} | +{(k-b)*100:.2f}pp")


if __name__ == "__main__":
    main()
