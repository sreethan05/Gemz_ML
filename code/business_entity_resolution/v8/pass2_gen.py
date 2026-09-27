"""Generic pass-2: rebuild a country pool + index, save candidates+35d feats
for given variant tags (for winners outside the default save set).
Usage: python pass2_gen.py <country> <tag1,tag2>
"""
import gc
import json
import pickle
import sys
import time
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "scratch" / "v8"
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4.core import EntityRecord, ensure_hashseed  # noqa: E402
from retrieval_v8 import (CAP, N_EVAL, build_csrs, build_pool, idf_pool,  # noqa: E402
                          load_truth, rescore_all, rescore_one, split_eval,
                          VARIANTS)
from feats8 import batch_pair_features, N_FEATURES_V8  # noqa: E402


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# deeper-retrieval extras: (tag, k0, cap, (wn, wa, wi, neutral))
EXTRAS = {
    "x1": ("x1", 1200, 50, VARIANTS["n4"]),
    "x2": ("x2", 1200, 40, VARIANTS["n4"]),
}


def main(country, tags):
    ensure_hashseed()
    for t in tags:
        assert t in VARIANTS, f"unknown variant {t}"
    tags = list(tags) + list(EXTRAS.keys())
    pool = build_pool(country)
    csrs = build_csrs(pool)
    ents = []
    with open(ROOT / "dataset" / "train" / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3].strip().lower() == country and split_eval(p[0]):
                ents.append((p[0], p[1], p[2]))
                if len(ents) >= N_EVAL:
                    break
    truth = load_truth({e[0] for e in ents})
    recs = [EntityRecord(e, nm, ad) for e, nm, ad in ents]
    del ents
    store = {t: ([], array("f")) for t in tags}
    hits = {t: 0 for t in tags}
    dens = 0
    t0 = time.time()
    for i, r in enumerate(recs):
        js, ss = idf_pool(csrs, r)
        if len(js):
            order = np.argsort(js)
            js = js[order]
            ss = ss[order]
            names = [pool.get_name(int(j)) for j in js]
            addrs = [pool.get_addr(int(j)) for j in js]
        else:
            names = addrs = []
        tops = rescore_all(js, ss, r, names, addrs, CAP)
        tr = set(truth.get(r.id, []))
        dens += len(tr)
        for t in tags:
            if t in EXTRAS:
                _, k0x, capx, (wnx, wax, wix, nex) = EXTRAS[t]
                jx, sx = idf_pool(csrs, r, k0=k0x)
                if len(jx):
                    ox = np.argsort(jx)
                    jx = jx[ox]
                    sx = sx[ox]
                    nx = [pool.get_name(int(j)) for j in jx]
                    ax = [pool.get_addr(int(j)) for j in jx]
                    top = rescore_one(jx, sx, r, nx, ax, capx, wnx, wax, wix, nex)
                else:
                    top = jx
            else:
                top = tops[t]
            top = np.sort(top)
            cand_ids, feats_all = store[t]
            cand_ids.append([pool.get_id(int(j)) for j in top])
            if len(top):
                cands = [pool.record(int(j)) for j in top]
                feats_all.extend(batch_pair_features(r, cands).ravel())
            hits[t] += len(tr & {pool.get_id(int(j)) for j in top})
        if (i + 1) % 10000 == 0:
            log(f"{country} pass2 {i+1:,}/{len(recs):,} ({time.time()-t0:.0f}s)")
    for t in tags:
        cand_ids, feats_all = store[t]
        with open(SCR / f"{country}_candids_{t}.pkl", "wb") as f:
            pickle.dump(cand_ids, f)
        X = np.frombuffer(feats_all, dtype=np.float32).reshape(-1, N_FEATURES_V8)
        np.save(SCR / f"{country}_feats_{t}.npy", X.copy())
        log(f"{country} saved {t}: {X.shape[0]:,} pairs "
            f"recall={hits[t]/max(dens,1):.4f}")
    del store, recs
    gc.collect()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
