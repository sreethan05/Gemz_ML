"""v8 cloud inference: multicore full-test inference for Linux VMs.

Same retrieval/feature/decision logic as run_test_v8.py, but entity batches
are distributed over N worker processes (fork start method — Linux only).
Run:  python cloud_infer.py <country> <n_workers> [--test-dir dataset/test]
Outputs: output/matching_results_v8_<country>.tsv (+ _cand + claims sidecar)
Assumes artifacts exist: models/lgbm_model_v8_m1.pkl, scratch/v8/<c>_retrieval_ab.json,
scratch/v8/variant_params.json, scratch/v8/<c>_best_rule.json.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
sys.path.insert(0, str(ROOT / "scratch" / "v8"))

from v4.core import PackedPool, CSRIndex, EntityRecord, ensure_hashseed  # noqa: E402
from v4.core import keys_of, key_hashes  # noqa: E402
from feats8 import batch_pair_features, N_FEATURES_V8  # noqa: E402
from scratch.antigravity.phonetic import phonetic_collapse  # noqa: E402

CSR_CHUNK = 10_000_000
K0 = 600
CAP = 40

_G = {}  # per-worker globals


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def build_pool_file(path: Path) -> PackedPool:
    pool = PackedPool()
    pool.cache_cap = 1
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 3:
                pool.add_raw(p[0], p[1], p[2])
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = 200_000
    log(f"pool {path.name}: {pool.n:,} records")
    return pool


def build_csrs(pool):
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        csrs.append(CSRIndex.build(pool, j0, min(j0 + CSR_CHUNK, pool.n),
                                   max_df=3000))
        pool._cache.clear()
    log(f"  index ready: {sum(len(c.J) for c in csrs):,} postings")
    return csrs


def w_init(country, variant_tag, rule):
    from v4.core import PackedPool as _P  # noqa
    _G["country"] = country
    _G["rule"] = rule
    import variant_params
    _G["rcfg"] = json.load(open(ROOT / "scratch" / "v8" / "variant_params.json"))[variant_tag]
    _G["clf"] = pickle.load(open(ROOT / "models" / "lgbm_model_v8_m1.pkl", "rb"))
    _G["clf"].set_params(n_jobs=1)


def w_process(chunk_lines):
    pool, csrs = _G["pool"], _G["csrs"]
    wn, wa, wi, neutral = _G["rcfg"]["wn"], _G["rcfg"]["wa"], _G["rcfg"]["wi"], _G["rcfg"]["neutral"]
    th, mt = _G["rule"]["th"], _G["rule"]["mt"]
    cap = _G["rcfg"].get("cap", CAP)
    from rapidfuzz import fuzz, process as rfp
    out_m, out_c, out_cl = [], [], []
    for line in chunk_lines:
        p = line.rstrip("\r\n").split("\t")
        if len(p) < 3:
            continue
        r = EntityRecord(p[0], p[1], p[2])
        js, ss = idf_pool(csrs, r)
        if len(js):
            order = np.argsort(js)
            js = js[order]
            ss = ss[order]
            names = [pool.get_name(int(j)) for j in js]
            addrs = [pool.get_addr(int(j)) for j in js]
            idf_n = ss / max(float(ss.max()), 1e-9)
            n_sim = rfp.cdist([r.name_norm], names, scorer=fuzz.token_set_ratio,
                              dtype=np.uint8)[0] / 100.0
            if bool(r.addr_norm):
                cand_has = np.fromiter((bool(a) for a in addrs), dtype=bool,
                                       count=len(addrs))
                a_raw = rfp.cdist([r.addr_norm], addrs, scorer=fuzz.token_set_ratio,
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
            top = np.sort(js[sel])
            cids = [pool.get_id(int(j)) for j in top]
            cands = [pool.record(int(j)) for j in top]
            probs = _G["clf"].predict_proba(batch_pair_features(r, cands))[:, 1]
            resc = _G["rule"].get("rescue")
            rescue_acc = []
            if resc and "lo" in resc and "tau" in resc:
                band = [i for i in range(len(cids)) if resc["lo"] <= probs[i] < th]
                qc = phonetic_collapse(r.name_norm)
                if band and qc:
                    phs = rfp.cdist(
                        [qc], [phonetic_collapse(c.name_norm) for c in cands],
                        scorer=fuzz.token_set_ratio)[0] / 100
                    rescue_acc = [i for i in band if phs[i] >= resc["tau"]]
            out_c.append(r.id + "\t" + ",".join(cids) + "\n")
            best_p = float(probs.max())
            if best_p < mt and not rescue_acc:
                out_m.append(r.id + "\t\n")
            else:
                acc = [i for i in range(len(cids)) if probs[i] >= th]
                acc = sorted(set(acc + rescue_acc))
                if not acc:
                    out_m.append(r.id + "\t\n")
                else:
                    out_m.append(r.id + "\t" + ",".join(sorted(cids[i] for i in acc)) + "\n")
                    for i in acc:
                        out_cl.append(f"{r.id}\t{cids[i]}\t{probs[i]:.6f}\n")
        else:
            out_m.append(r.id + "\t\n")
            out_c.append(r.id + "\t\n")
    return "".join(out_m), "".join(out_c), "".join(out_cl)


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


def run_shard(country, k, n):
    """Sequential worker: processes a contiguous slice of S1 (shard k of n)."""
    ensure_hashseed()
    with open(ROOT / "scratch" / "v8" / f"{country}_best_rule.json") as f:
        rule = json.load(f)
    _G["rule"] = rule
    _G["rcfg"] = json.load(open(ROOT / "scratch" / "v8" / "variant_params.json"))[rule["tag"]]
    _G["clf"] = load_model(rule)
    pool, csrs = get_pool_csrs(country)
    _G["pool"], _G["csrs"] = pool, csrs
    sys.path.insert(0, str(ROOT / "scratch" / "v8"))
    all_lines = []
    with open(ROOT / "dataset" / "test" / "test_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("
").split("	")
            if len(p) >= 4 and p[3].strip().lower() == country:
                all_lines.append(line)
    total = len(all_lines)
    per = (total + n - 1) // n
    lo, hi = k * per, min((k + 1) * per, total)
    lines = all_lines[lo:hi]
    del all_lines
    log(f"shard {k}/{n}: {len(lines):,} entities [{lo:,}:{hi:,})")
    t0 = time.time()
    m_path = ROOT / "output" / f"mr_v8_{country}_s{k}.tsv"
    c_path = ROOT / "output" / f"cp_v8_{country}_s{k}.tsv"
    cl_path = ROOT / "scratch" / "v8" / f"claims_v8_{country}_s{k}.tsv"
    with open(m_path, "w", encoding="utf-8", newline="\n") as fm, \
         open(c_path, "w", encoding="utf-8", newline="\n") as fc, \
         open(cl_path, "w", encoding="utf-8", newline="\n") as fcl:
        res = w_process(lines)
        fm.write(res[0])
        fc.write(res[1])
        fcl.write(res[2])
    log(f"shard {k}/{n} done in {(time.time()-t0)/60:.1f} min")


def build_pool_country(country):
    pool = PackedPool()
    pool.cache_cap = 1
    t0 = time.time()
    for fn in ("test_source2.tsv", "test_source3.tsv"):
        with open(ROOT / "dataset" / "test" / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("
").split("	")
                if len(p) >= 4 and p[3].strip().lower() == country:
                    pool.add_raw(p[0], p[1], p[2])
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = 200_000
    log(f"{country} pool: {pool.n:,} records in {time.time()-t0:.0f}s")
    return pool


def build_csrs_country(pool):
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        csrs.append(CSRIndex.build(pool, j0, min(j0 + CSR_CHUNK, pool.n), max_df=3000))
        pool._cache.clear()
    log(f"  index ready: {sum(len(c.J) for c in csrs):,} postings")
    return csrs


def load_model(rule):
    """Load the calibrated model (m1 / m2 / probability-averaged ensemble)."""
    if rule["model"] == "ens":
        m1 = pickle.load(open(ROOT / "models" / "lgbm_model_v8_m1.pkl", "rb"))
        m2 = pickle.load(open(ROOT / "models" / "lgbm_model_v8_m2.pkl", "rb"))
        m1.set_params(n_jobs=1)
        m2.set_params(n_jobs=1)
        class Ens:
            @staticmethod
            def predict_proba(X):
                p = (m1.predict_proba(X)[:, 1] + m2.predict_proba(X)[:, 1]) / 2
                return np.column_stack((1 - p, p))
        return Ens()
    return pickle.load(open(ROOT / f"models/lgbm_model_v8_{rule['model']}.pkl", "rb"))


def get_pool_csrs(country):
    """Pool+CSR with on-disk cache (build once per machine, seed-consistent)."""
    cdir = ROOT / "cloud_cache" / country
    if (cdir / "pool.pkl").exists() and (cdir / "csrs.pkl").exists():
        log(f"{country}: loading cached pool+index")
        pool = pickle.load(open(cdir / "pool.pkl", "rb"))
        csrs = pickle.load(open(cdir / "csrs.pkl", "rb"))
        return pool, csrs
    pool = build_pool_file(country)
    csrs = build_csrs(pool)
    cdir.mkdir(parents=True, exist_ok=True)
    pickle.dump(pool, open(cdir / "pool.pkl", "wb"), protocol=4)
    pickle.dump(csrs, open(cdir / "csrs.pkl", "wb"), protocol=4)
    log(f"{country}: cached pool+index")
    return pool, csrs


def main():
    ensure_hashseed()
    country = sys.argv[1]
    if "--shard" in sys.argv:
        k = int(sys.argv[sys.argv.index("--shard") + 1])
        n = int(sys.argv[sys.argv.index("--shard") + 2])
        run_shard(country, k, n)
        return
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with open(ROOT / "scratch" / "v8" / f"{country}_retrieval_ab.json") as f:
        best = json.load(f)["best"]
    with open(ROOT / "scratch" / "v8" / f"{country}_best_rule.json") as f:
        rule = json.load(f)

    t00 = time.time()
    pool = build_pool_file(ROOT / "dataset" / "test" / "partitions" / f"{country}_other.tsv")
    csrs = build_csrs(pool)
    # share pool/csrs via module globals inherited by fork
    _G["pool"] = pool
    _G["csrs"] = csrs
    _G["rcfg"] = json.load(open(ROOT / "scratch" / "v8" / "variant_params.json"))[best]
    _G["clf"] = pickle.load(open(ROOT / "models" / "lgbm_model_v8_m1.pkl", "rb"))
    _G["clf"].set_params(n_jobs=1)
    _G["rule"] = rule

    s1_path = ROOT / "dataset" / "test" / "partitions" / f"{country}_s1.tsv"
    with open(s1_path, encoding="utf-8") as f:
        next(f)
        lines = f.readlines()
    log(f"{country}: {len(lines):,} S1 entities, {n_workers} workers")
    CH = 2000
    chunks = [lines[i:i + CH] for i in range(0, len(lines), CH)]
    del lines

    m_path = ROOT / "output" / f"matching_results_v8_{country}.tsv"
    c_path = ROOT / "output" / f"candidate_pairs_v8_{country}.tsv"
    cl_path = ROOT / "scratch" / "v8" / f"claims_v8_{country}.tsv"
    n_done = 0
    with Pool(n_workers) as wp, \
         open(m_path, "w", encoding="utf-8", newline="\n") as fm, \
         open(c_path, "w", encoding="utf-8", newline="\n") as fc, \
         open(cl_path, "w", encoding="utf-8", newline="\n") as fcl:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for om, oc, ocl in wp.imap_unordered(_w_process_safe, chunks, chunksize=1):
            fm.write(om)
            fc.write(oc)
            fcl.write(ocl)
            n_done += CH
            if n_done % 100_000 < CH:
                log(f"  {n_done:,}/{len(chunks)*CH:,} ({time.time()-t00:.0f}s)")
    log(f"{country} inference done in {(time.time()-t00)/60:.1f} min")


def _w_process_safe(chunk):
    try:
        return w_process(chunk)
    except Exception as e:  # noqa
        import traceback
        return ("", "", f"WARN\t{chunk[0][:20] if chunk else '?'}\t{traceback.format_exc()[-500:]}\n")


if __name__ == "__main__":
    main()
