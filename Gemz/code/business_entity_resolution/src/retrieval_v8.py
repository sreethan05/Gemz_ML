"""v8 Phase B final: streaming pool + rescore A/B + train-pair mining.

Memory-safe streaming: per entity, the k0-pool is processed immediately for
all variants (nothing accumulated). One shared cdist per entity feeds all
variant weightings. Outputs per country in scratch/v8/:
  {c}_retrieval_ab.json, {c}_candids_{tag}.pkl, {c}_feats_{tag}.npy (35d),
  {c}_eval_meta.pkl, {c}_train_X.npy, {c}_train_y.npy
"""
import gc
import hashlib
import json
import pickle
import sys
import time
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "dataset" / "test").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
while not (ROOT / 'dataset' / 'test').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
CONFIGS = ROOT / "scratch" / "v8"
if not CONFIGS.exists():
    CONFIGS = Path(__file__).resolve().parent / "configs"
while not (ROOT / 'dataset' / 'test').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

TRAIN = ROOT / "dataset" / "train"
SCR = CONFIGS
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (PackedPool, CSRIndex, EntityRecord, ensure_hashseed,  # noqa: E402
                     keys_of, key_hashes)
from feats8 import batch_pair_features, N_FEATURES_V8  # noqa: E402

SCR.mkdir(exist_ok=True)
CSR_CHUNK = 10_000_000
N_EVAL = 25_000
N_TRAIN_MINE = 100_000
CAP = 40
K0 = 600

# tag -> (wn, wa, wi, neutral); all at k0=K0. neutral=addr component used when
# either side has an empty address (missing != contradicting).
VARIANTS = {
    "n1": (0.35, 0.45, 0.20, 0.55),
    "n2": (0.45, 0.35, 0.20, 0.55),
    "n3": (0.55, 0.25, 0.20, 0.55),
    "n4": (0.35, 0.55, 0.10, 0.55),
    "n5": (0.45, 0.45, 0.10, 0.55),
}
SAVE_TAGS = ("n1", "n2", "n3", "n4", "n5")  # feats pass saves top-3 by recall (runtime)


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def split_eval(eid: str) -> bool:
    return int(hashlib.md5(eid.encode()).hexdigest()[:8], 16) % 1000 >= 960


def build_pool(country: str) -> PackedPool:
    pool = PackedPool()
    pool.cache_cap = 1
    t0 = time.time()
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4 and p[3].strip().lower() == country:
                    pool.add_raw(p[0], p[1], p[2])
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = 100_000
    log(f"{country} pool: {pool.n:,} records in {time.time()-t0:.0f}s")
    return pool


def build_csrs(pool):
    # max_df=3000 == query-time max_df_query: keys with df>3000 are never used
    # by queries, so dropping them at build is a zero-semantic-change memory
    # cut (postings 209M -> ~90M on US; no page-thrash on the 8GB box).
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        csrs.append(CSRIndex.build(pool, j0, min(j0 + CSR_CHUNK, pool.n),
                                   max_df=3000))
        pool._cache.clear()
        gc.collect()
    log(f"  index ready: {sum(len(c.J) for c in csrs):,} postings")
    return csrs


def idf_pool(csrs, r, k0=K0):
    per = max(k0 // max(len(csrs), 1) + 20, 60)
    keys = keys_of(r)
    hs, ws, mdfs = key_hashes(keys)
    all_j, all_s = [], []
    for csr in csrs:
        js, ss = csr.query(hs, ws, per, mdfs)
        if js is not None:
            all_j.append(js)
            all_s.append(ss)
    if not all_j:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    js = np.concatenate(all_j)
    ss = np.concatenate(all_s)
    js, first = np.unique(js, return_index=True)
    ss = ss[first]
    if len(js) > k0:
        sel = np.argpartition(ss, -k0)[-k0:]
        js, ss = js[sel], ss[sel]
    return js, ss


def rescore_all(js, ss, r, names, addrs, cap):
    """All variant top-caps from one pre-fetched pool; shared cdist per entity."""
    from rapidfuzz import fuzz as _fuzz, process as _proc
    if len(js) == 0:
        return {tag: js for tag in VARIANTS}
    idf_n = ss / max(float(ss.max()), 1e-9)
    n_sim = _proc.cdist([r.name_norm], names, scorer=_fuzz.token_set_ratio,
                        dtype=np.uint8)[0] / 100.0
    q_has_a = bool(r.addr_norm)
    a_raw = None
    cand_has = None
    if q_has_a:
        cand_has = np.fromiter((bool(a) for a in addrs), dtype=bool, count=len(addrs))
        a_raw = _proc.cdist([r.addr_norm], addrs, scorer=_fuzz.token_set_ratio,
                            dtype=np.uint8)[0] / 100.0
    out = {}
    for tag, (wn, wa, wi, neutral) in VARIANTS.items():
        if q_has_a:
            a_eff = np.where(cand_has, a_raw, neutral).astype(np.float32)
        else:
            a_eff = np.full(len(js), neutral, dtype=np.float32)
        combined = wn * n_sim + wa * a_eff + wi * idf_n
        if len(js) > cap:
            sel = np.argpartition(combined, -cap)[-cap:]
            sel = sel[np.argsort(combined[sel])[::-1]]
        else:
            sel = np.argsort(combined)[::-1]
        out[tag] = np.sort(js[sel])
    return out


def rescore_one(js, ss, r, names, addrs, cap, wn, wa, wi, neutral):
    from rapidfuzz import fuzz as _fuzz, process as _proc
    if len(js) == 0:
        return js
    idf_n = ss / max(float(ss.max()), 1e-9)
    n_sim = _proc.cdist([r.name_norm], names, scorer=_fuzz.token_set_ratio,
                        dtype=np.uint8)[0] / 100.0
    q_has_a = bool(r.addr_norm)
    if q_has_a:
        cand_has = np.fromiter((bool(a) for a in addrs), dtype=bool, count=len(addrs))
        a_raw = _proc.cdist([r.addr_norm], addrs, scorer=_fuzz.token_set_ratio,
                            dtype=np.uint8)[0] / 100.0
        a_eff = np.where(cand_has, a_raw, neutral).astype(np.float32)
    else:
        a_eff = np.full(len(js), neutral, dtype=np.float32)
    combined = wn * n_sim + wa * a_eff + wi * idf_n
    if len(js) > cap:
        sel = np.argpartition(combined, -cap)[-cap:]
        sel = sel[np.argsort(combined[sel])[::-1]]
    else:
        sel = np.argsort(combined)[::-1]
    return np.sort(js[sel])


def load_truth(want_ids):
    truth = {}
    with open(TRAIN / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            if s1 in want_ids:
                truth[s1] = [t for t in rest.split(",") if t]
    return truth


def eval_pass(country, pool, csrs):
    ents = []
    with open(TRAIN / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3].strip().lower() == country and split_eval(p[0]):
                ents.append((p[0], p[1], p[2]))
                if len(ents) >= N_EVAL:
                    break
    log(f"{country}: {len(ents):,} eval entities")
    truth = load_truth({e[0] for e in ents})
    recs = [EntityRecord(e, nm, ad) for e, nm, ad in ents]
    del ents

    t0 = time.time()
    hits = {tag: 0 for tag in VARIANTS}
    dens = 0
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
        for tag in VARIANTS:
            hits[tag] += len(tr & {pool.get_id(int(j)) for j in tops[tag]})
        if (i + 1) % 5000 == 0:
            log(f"  eval-recall {i+1:,}/{len(recs):,} ({time.time()-t0:.0f}s)")

    results = {tag: hits[tag] / max(dens, 1) for tag in VARIANTS}
    for tag, pr in results.items():
        log(f"{country} [{tag}]: pair-recall@{CAP}={pr:.4f}")
    best = max(results, key=results.get)
    log(f"{country} BEST variant: {best} ({results[best]:.4f})")
    with open(SCR / f"{country}_retrieval_ab.json", "w") as f:
        json.dump({"results": results, "best": best}, f, indent=1)

    # pass 2: candidates + features for the top-3 variants by recall
    top3 = sorted(results, key=results.get, reverse=True)[:3]
    log(f"{country} feats will be saved for: {top3}")
    store = {tag: ([], array("f")) for tag in top3}
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
        for tag in top3:
            top = tops[tag]
            cand_ids, feats_all = store[tag]
            cand_ids.append([pool.get_id(int(j)) for j in top])
            if len(top):
                cands = [pool.record(int(j)) for j in top]
                feats_all.extend(batch_pair_features(r, cands).ravel())
        if (i + 1) % 5000 == 0:
            log(f"  eval-feats {i+1:,}/{len(recs):,} ({time.time()-t0:.0f}s)")
    eval_recs_meta = [(r.id, r.name_norm, r.addr_norm) for r in recs]
    del recs
    gc.collect()

    with open(SCR / f"{country}_eval_meta.pkl", "wb") as f:
        pickle.dump({"eval_recs": eval_recs_meta, "eval_truth": truth}, f)
    for tag in top3:
        cand_ids, feats_all = store[tag]
        with open(SCR / f"{country}_candids_{tag}.pkl", "wb") as f:
            pickle.dump(cand_ids, f)
        if len(feats_all):
            X = np.frombuffer(feats_all, dtype=np.float32).reshape(-1, N_FEATURES_V8)
            np.save(SCR / f"{country}_feats_{tag}.npy", X.copy())
    del store
    gc.collect()
    return best


def mine_pass(country, pool, csrs, best_tag):
    wn, wa, wi, neutral = VARIANTS[best_tag]
    ents = []
    with open(TRAIN / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3].strip().lower() == country and not split_eval(p[0]):
                ents.append((p[0], p[1], p[2]))
                if len(ents) >= N_TRAIN_MINE:
                    break
    log(f"{country}: mining {len(ents):,} train entities with [{best_tag}]")
    truth = load_truth({e[0] for e in ents})
    t0 = time.time()
    X = np.empty((len(ents) * CAP, N_FEATURES_V8), dtype=np.float32)
    y = np.empty(len(ents) * CAP, dtype=np.int8)
    n = 0
    for i, (eid, nm, ad) in enumerate(ents):
        r = EntityRecord(eid, nm, ad)
        js, ss = idf_pool(csrs, r)
        if len(js):
            order = np.argsort(js)
            js = js[order]
            ss = ss[order]
            names = [pool.get_name(int(j)) for j in js]
            addrs = [pool.get_addr(int(j)) for j in js]
            top = rescore_one(js, ss, r, names, addrs, CAP, wn, wa, wi, neutral)
            if len(top):
                cands = [pool.record(int(j)) for j in top]
                F = batch_pair_features(r, cands)
                trset = set(truth.get(eid, []))
                m = len(top)
                X[n:n + m] = F
                y[n:n + m] = np.fromiter((pool.get_id(int(j)) in trset for j in top),
                                         dtype=np.int8, count=m)
                n += m
        if (i + 1) % 10_000 == 0:
            log(f"  mined {i+1:,}/{len(ents):,} rows={n:,} pos={int(y[:n].sum()):,} "
                f"({time.time()-t0:.0f}s)")
    X = X[:n]
    y = y[:n]
    np.save(SCR / f"{country}_train_X.npy", X)
    np.save(SCR / f"{country}_train_y.npy", y)
    log(f"{country} mined: {n:,} pairs, {int(y.sum()):,} positive "
        f"({y.mean():.2%}) in {time.time()-t0:.0f}s")


def main(country):
    ensure_hashseed()
    t00 = time.time()
    pool = build_pool(country)
    csrs = build_csrs(pool)
    best = eval_pass(country, pool, csrs)
    mine_pass(country, pool, csrs, best)
    log(f"DONE {country} in {(time.time()-t00)/60:.1f} min (best={best})")


if __name__ == "__main__":
    main(sys.argv[1])
