"""v8 cloud inference: multicore full-test inference for Linux VMs.

Shard mode (used now): N independent single-process shards, each loading the
cached pool+index and processing a contiguous slice of S1 entities. Runs
reliably over ssh (no multiprocessing inside one process).
Run:  PYTHONHASHSEED=1 python cloud_infer.py <country> --shard <k> <n>
Outputs per shard: output/mr_v8_<country>_s<k>.tsv (+ candidate + claims).
"""
import json
import pickle
import sys
import time
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CONFIGS))
sys.path.insert(0, str(ROOT / "scratch" / "antigravity"))

from core import PackedPool, CSRIndex, EntityRecord, ensure_hashseed  # noqa: E402
from core import keys_of, key_hashes  # noqa: E402
from feats8 import batch_pair_features, N_FEATURES_V8  # noqa: E402
from phonetic import phonetic_collapse  # noqa: E402

CSR_CHUNK = 10_000_000
K0_DEFAULT = 600
CAP_DEFAULT = 40
NL = chr(10)

_G = {}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def build_pool_country(country):
    pool = PackedPool()
    pool.cache_cap = 1
    t0 = time.time()
    for fn in ("test_source2.tsv", "test_source3.tsv"):
        with open(ROOT / "dataset" / "test" / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
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


def get_pool_csrs(country):
    """Pool+CSR with on-disk cache (build once per machine, seed-consistent)."""
    cdir = ROOT / "cloud_cache" / country
    if (cdir / "pool.pkl").exists() and (cdir / "csrs.pkl").exists():
        log(f"{country}: loading cached pool+index")
        pool = pickle.load(open(cdir / "pool.pkl", "rb"))
        csrs = pickle.load(open(cdir / "csrs.pkl", "rb"))
        return pool, csrs
    pool = build_pool_country(country)
    csrs = build_csrs_country(pool)
    cdir.mkdir(parents=True, exist_ok=True)
    pickle.dump(pool, open(cdir / "pool.pkl", "wb"), protocol=4)
    pickle.dump(csrs, open(cdir / "csrs.pkl", "wb"), protocol=4)
    log(f"{country}: cached pool+index")
    return pool, csrs


def load_model(rule):
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


def stream_s1(country):
    with open(ROOT / "dataset" / "test" / "test_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3].strip().lower() == country:
                yield line


def idf_pool(csrs, r, k0):
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


def w_process(chunk_lines, pool, csrs, rcfg, rule, clf):
    from rapidfuzz import fuzz, process as rfp
    cap = int(rcfg.get("cap", CAP_DEFAULT))
    k0 = int(rcfg.get("k0", K0_DEFAULT))
    wn, wa, wi = rcfg["wn"], rcfg["wa"], rcfg["wi"]
    neutral = rcfg["neutral"]
    wp_boost = rcfg.get("wp")
    th, mt = rule["th"], rule["mt"]
    resc = rule.get("rescue")
    out_m, out_c, out_cl = [], [], []
    for line in chunk_lines:
        p = line.rstrip("\r\n").split("\t")
        if len(p) < 3:
            continue
        r = EntityRecord(p[0], p[1], p[2])
        js, ss = idf_pool(csrs, r, k0)
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
            if wp_boost and bool(r.name_norm):
                qc = phonetic_collapse(r.name_norm)
                if qc:
                    ph = rfp.cdist([qc], [phonetic_collapse(n) for n in names],
                                   scorer=fuzz.token_set_ratio,
                                   dtype=np.uint8)[0] / 100.0
                    combined = combined + wp_boost * ph
            if len(js) > cap:
                sel = np.argpartition(combined, -cap)[-cap:]
                sel = sel[np.argsort(combined[sel])[::-1]]
            else:
                sel = np.argsort(combined)[::-1]
            top = np.sort(js[sel])
            cids = [pool.get_id(int(j)) for j in top]
            cands = [pool.record(int(j)) for j in top]
            probs = clf.predict_proba(batch_pair_features(r, cands))[:, 1]
            out_c.append(r.id + "\t" + ",".join(cids) + "\n")
            best_p = float(probs.max())
            if best_p < mt:
                out_m.append(r.id + "\t" + NL)
            else:
                acc = [i for i in range(len(cids)) if probs[i] >= th]
                if resc and bool(r.name_norm):
                    lo, tau = resc["lo"], resc["tau"]
                    qc = phonetic_collapse(r.name_norm)
                    if qc:
                        band = [i for i in range(len(cids)) if lo <= probs[i] < th]
                        if band:
                            phs = rfp.cdist(
                                [qc],
                                [phonetic_collapse(cands[i].name_norm) for i in band],
                                scorer=fuzz.token_set_ratio,
                                dtype=np.uint8)[0] / 100.0
                            acc += [i for i, b in zip(band, phs) if b >= tau]
                acc = sorted(set(acc))
                if not acc:
                    out_m.append(r.id + "\t" + NL)
                else:
                    out_m.append(r.id + "\t" + ",".join(sorted(cids[i] for i in acc)) + "\n")
                    for i in acc:
                        out_cl.append(f"{r.id}\t{cids[i]}\t{probs[i]:.6f}\n")
        else:
            out_m.append(r.id + "\t" + NL)
            out_c.append(r.id + "\t" + NL)
    return "".join(out_m), "".join(out_c), "".join(out_cl)


def run_shard(country, k, n):
    ensure_hashseed()
    with open(CONFIGS / f"{country}_best_rule.json", encoding="utf-8") as f:
        rule = json.load(f)
    rcfg = json.load(open(CONFIGS / "variant_params.json"))[rule["tag"]]
    clf = load_model(rule)
    pool, csrs = get_pool_csrs(country)
    all_lines = list(stream_s1(country))
    total = len(all_lines)
    per = (total + n - 1) // n
    lo, hi = k * per, min((k + 1) * per, total)
    lines = all_lines[lo:hi]
    del all_lines
    log(f"shard {k}/{n}: {len(lines):,} entities [{lo:,}:{hi:,})")
    t0 = time.time()
    m_path = ROOT / "output" / f"mr_v8_{country}_s{k}.tsv"
    c_path = ROOT / "output" / f"cp_v8_{country}_s{k}.tsv"
    cl_path = CONFIGS / f"claims_v8_{country}_s{k}.tsv"
    res = w_process(lines, pool, csrs, rcfg, rule, clf)
    with open(m_path, "w", encoding="utf-8", newline=NL) as fm:
        fm.write(res[0])
    with open(c_path, "w", encoding="utf-8", newline=NL) as fc:
        fc.write(res[1])
    with open(cl_path, "w", encoding="utf-8", newline=NL) as fcl:
        fcl.write(res[2])
    log(f"shard {k}/{n} done in {(time.time()-t0)/60:.1f} min")


def run_range(country, lo, hi, tag):
    ensure_hashseed()
    with open(CONFIGS / f"{country}_best_rule.json", encoding="utf-8") as f:
        rule = json.load(f)
    rcfg = json.load(open(CONFIGS / "variant_params.json"))[rule["tag"]]
    clf = load_model(rule)
    pool, csrs = get_pool_csrs(country)
    all_lines = list(stream_s1(country))
    lines = all_lines[lo:hi]
    del all_lines
    log(f"range [{lo:,}:{hi:,}): {len(lines):,} entities")
    t0 = time.time()
    m_path = ROOT / "output" / f"mr_v8_{country}_{tag}.tsv"
    c_path = ROOT / "output" / f"cp_v8_{country}_{tag}.tsv"
    res = w_process(lines, pool, csrs, rcfg, rule, clf)
    with open(m_path, "w", encoding="utf-8", newline=NL) as fm:
        fm.write(res[0])
    with open(c_path, "w", encoding="utf-8", newline=NL) as fc:
        fc.write(res[1])
    log(f"range [{lo:,}:{hi:,}) done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ensure_hashseed()
    country = sys.argv[1]
    if "--range" in sys.argv:
        i = sys.argv.index("--range")
        lo, hi, tag = int(sys.argv[i+1]), int(sys.argv[i+2]), sys.argv[i+3]
        run_range(country, lo, hi, tag)
        sys.exit(0)
    k = int(sys.argv[sys.argv.index("--shard") + 1])
    n = int(sys.argv[sys.argv.index("--shard") + 2])
    run_shard(country, k, n)
