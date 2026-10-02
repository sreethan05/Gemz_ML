"""v8 Phase D: production inference over the full test set.

Per country: pool from dataset/test/partitions/{c}_other.tsv, CSR index,
k0=600 IDF retrieval, winner-rescore (from scratch/v8/{c}_retrieval_ab.json),
35-dim batch features, LightGBM v8 model, calibrated (th, mt) rule from
scratch/v8/{c}_best_rule.json, claims sidecar + exact 1-to-1 prune.

Writes output/matching_results_v8_{c}.tsv + candidate_pairs_v8_{c}.tsv
(never touches the board files), then a merge step produces
output/matching_results_v8.tsv (+_candids) and runs the official validator.
"""
import gc
import json
import pickle
import subprocess
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

PARTS = ROOT / "dataset" / "test" / "partitions"
OUTD = ROOT / "output"
SCR8 = CONFIGS
MODELS = ROOT / "models"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (PackedPool, CSRIndex, EntityRecord, ensure_hashseed,  # noqa: E402
                     keys_of, key_hashes)
from feats8 import batch_pair_features, N_FEATURES_V8  # noqa: E402

BATCH = 4000
CSR_CHUNK = 10_000_000
K0 = 600
EXPECT = {"france": 259452, "us": 663106, "india": 809986}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def build_pool_file(path: Path) -> PackedPool:
    pool = PackedPool()
    pool.cache_cap = 1
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 3:
                pool.add_raw(p[0], p[1], p[2])
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = 100_000
    log(f"pool {path.name}: {pool.n:,} records in {time.time()-t0:.0f}s")
    return pool


def build_csrs(pool):
    # max_df=3000 == query-time max_df_query (zero semantic change; see v8 notes)
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        csrs.append(CSRIndex.build(pool, j0, min(j0 + CSR_CHUNK, pool.n),
                                   max_df=3000))
        pool._cache.clear()
        gc.collect()
    log(f"  index ready: {sum(len(c.J) for c in csrs):,} postings")
    return csrs


def stream_s1(path: Path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 3:
                yield EntityRecord(p[0], p[1], p[2])


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


def rescore_top(js, ss, r, names, addrs, cap, wn, wa, wi, neutral):
    from rapidfuzz import fuzz as _fuzz, process as _proc
    if len(js) == 0:
        return js
    idf_n = ss / max(float(ss.max()), 1e-9)
    n_sim = _proc.cdist([r.name_norm], names, scorer=_fuzz.token_set_ratio,
                        dtype=np.uint8)[0] / 100.0
    if neutral is None:
        a_sim = _proc.cdist([r.addr_norm], addrs, scorer=_fuzz.token_set_ratio,
                            dtype=np.uint8)[0] / 100.0
        combined = wn * n_sim + wa * a_sim + wi * idf_n
    else:
        q_has_a = bool(r.addr_norm)
        if q_has_a:
            cand_has = np.fromiter((bool(a) for a in addrs), dtype=bool,
                                   count=len(addrs))
            a_sim = _proc.cdist([r.addr_norm], addrs,
                                scorer=_fuzz.token_set_ratio,
                                dtype=np.uint8)[0] / 100.0
            a_eff = np.where(cand_has, a_sim, neutral).astype(np.float32)
        else:
            a_eff = np.full(len(js), neutral, dtype=np.float32)
        combined = wn * n_sim + wa * a_eff + wi * idf_n
    if len(js) > cap:
        sel = np.argpartition(combined, -cap)[-cap:]
        sel = sel[np.argsort(combined[sel])[::-1]]
    else:
        sel = np.argsort(combined)[::-1]
    return js[sel]


def run_country(ctry, rcfg, rule, clf, fm, fc, fclaims):
    cap = int(rcfg.get("cap", 40))
    wn, wa, wi = rcfg["wn"], rcfg["wa"], rcfg["wi"]
    neutral = rcfg["neutral"]
    th, mt = rule["th"], rule["mt"]
    k0 = int(rcfg.get("k0", K0))
    log(f"==== {ctry.upper()} (cap={cap} k0={k0} w=({wn},{wa},{wi}) "
        f"neutral={neutral} th={th} mt={mt}) ====")
    pool = build_pool_file(PARTS / f"{ctry}_other.tsv")
    csrs = build_csrs(pool)
    n_ent = n_single = n_matches = 0
    t0 = time.time()
    batch = []
    def flush(batch):
        nonlocal n_ent, n_single, n_matches
        if not batch:
            return
        cand_lists = []
        feat_flat = array("f")
        for r in batch:
            js, ss = idf_pool(csrs, r, k0)
            if len(js):
                order = np.argsort(js)
                js = js[order]
                ss = ss[order]
                names = [pool.get_name(int(j)) for j in js]
                addrs = [pool.get_addr(int(j)) for j in js]
                top = rescore_top(js, ss, r, names, addrs, cap, wn, wa, wi, neutral)
                top = np.sort(top)
                cand_lists.append([pool.get_id(int(j)) for j in top])
                cands = [pool.record(int(j)) for j in top]
                feat_flat.extend(batch_pair_features(r, cands).ravel())
            else:
                cand_lists.append([])
        n_pairs = len(feat_flat) // N_FEATURES_V8
        if n_pairs:
            X = np.frombuffer(feat_flat, dtype=np.float32).reshape(n_pairs, N_FEATURES_V8)
            probs_all = clf.predict_proba(X)[:, 1]
        else:
            probs_all = np.empty(0, dtype=np.float32)
        del feat_flat
        pos = 0
        for r, cids in zip(batch, cand_lists):
            n_ent += 1
            fc.write(r.id + "\t" + ",".join(cids) + "\n")
            if not cids:
                fm.write(r.id + "\t\n")
                n_single += 1
                continue
            k = len(cids)
            probs = probs_all[pos:pos + k]
            pos += k
            best_p = float(probs.max()) if k else -1.0
            if best_p < mt:
                fm.write(r.id + "\t\n")
                n_single += 1
                continue
            acc = [i for i in range(k) if probs[i] >= th]
            if not acc:
                fm.write(r.id + "\t\n")
                n_single += 1
                continue
            n_matches += len(acc)
            fm.write(r.id + "\t" + ",".join(sorted(cids[i] for i in acc)) + "\n")
            for i in acc:
                fclaims.write(f"{r.id}\t{cids[i]}\t{probs[i]:.6f}\n")
    for r in stream_s1(PARTS / f"{ctry}_s1.tsv"):
        batch.append(r)
        if len(batch) >= BATCH:
            flush(batch)
            batch = []
            if n_ent % 200_000 < BATCH:
                log(f"  {ctry}: {n_ent:,} entities ({time.time()-t0:.0f}s)")
    flush(batch)
    log(f"[{ctry}] entities={n_ent:,} singles={n_single:,} matches={n_matches:,} "
        f"in {(time.time()-t0)/60:.1f}min")
    return n_ent


def prune_1to1(ctry, match_path, claims_path):
    t0 = time.time()
    best = {}
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            s1, tid, pr = line.rstrip("\n").split("\t")
            pr = float(pr)
            cur = best.get(tid)
            if cur is None or pr > cur[0] or (pr == cur[0] and s1 < cur[1]):
                best[tid] = (pr, s1)
    tmp = match_path.with_suffix(f".{ctry}.tmp")
    kept = pruned = singles = rows = 0
    with open(match_path, encoding="utf-8") as f, \
         open(tmp, "w", encoding="utf-8", newline="\n") as g:
        g.write("source1_entity_id\tmatched_entity_ids\n")
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            mlist = [x for x in rest.split(",") if x] if rest else []
            kept_list = [t for t in mlist if best.get(t, (0.0, s1))[1] == s1]
            kept += len(kept_list)
            pruned += len(mlist) - len(kept_list)
            if not kept_list and mlist:
                singles += 1
            rows += 1
            g.write(s1 + "\t" + ",".join(sorted(kept_list)) + "\n")
    tmp.replace(match_path)
    log(f"[{ctry}] 1-to-1: kept {kept:,} pruned {pruned:,} new-singles {singles:,} "
        f"({time.time()-t0:.0f}s)")
    if rows != EXPECT.get(ctry):
        log(f"WARNING {ctry}: rows={rows:,} expected={EXPECT.get(ctry):,}")


def main():
    ensure_hashseed()
    ctry = sys.argv[1]
    with open(SCR8 / f"{ctry}_best_rule.json") as f:
        rule = json.load(f)
    # Phase C calibrates each model against each retrieval variant. Use that
    # exact winning pair at inference; choosing the retrieval-only winner or
    # always loading M1 silently invalidates the calibrated threshold.
    tag, model_name = rule["tag"], rule["model"]
    with open(SCR8 / "variant_params.json") as f:
        rcfg = json.load(f)[tag]
    if model_name in ("m1", "m2"):
        with open(MODELS / f"lgbm_model_v8_{model_name}.pkl", "rb") as f:
            clf = pickle.load(f)
        clf.set_params(n_jobs=1)
    elif model_name == "ens":
        with open(MODELS / "lgbm_model_v8_m1.pkl", "rb") as f:
            m1 = pickle.load(f)
        with open(MODELS / "lgbm_model_v8_m2.pkl", "rb") as f:
            m2 = pickle.load(f)
        m1.set_params(n_jobs=1)
        m2.set_params(n_jobs=1)

        class Ensemble:
            def predict_proba(self, X):
                p = (m1.predict_proba(X)[:, 1] + m2.predict_proba(X)[:, 1]) / 2
                return np.column_stack((1 - p, p))

        clf = Ensemble()
    else:
        raise ValueError(f"Unknown calibrated model {model_name!r}")
    m_path = OUTD / f"matching_results_v8_{ctry}.tsv"
    c_path = OUTD / f"candidate_pairs_v8_{ctry}.tsv"
    claims = SCR8 / f"claims_v8_{ctry}.tsv"
    with open(m_path, "w", encoding="utf-8", newline="\n") as fm, \
         open(c_path, "w", encoding="utf-8", newline="\n") as fc, \
         open(claims, "w", encoding="utf-8", newline="\n") as fcl:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        run_country(ctry, rcfg, rule, clf, fm, fc, fcl)
    prune_1to1(ctry, m_path, claims)
    log(f"DONE {ctry}")


if __name__ == "__main__":
    main()
