"""Measure CLUSTER-LEVEL acceptance on the expanded v6 eval artifacts.

Rule: group each entity's expanded candidates by sibling fingerprint; a
cluster is accepted if its best member's prob >= t; ALL members of accepted
clusters are predicted (they are 96.8% co-true). Sweep t and cluster cap.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import ensure_hashseed, EntityRecord, fingerprint_of, entity_f05

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "scratch" / "v4" / "v5_eval_backup"
MODELS = ROOT / "models"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def analyze(country: str, probs, cand_ids, eval_recs, eval_truth):
    # fingerprints for every expanded candidate id
    need = set()
    for ids in cand_ids:
        need.update(ids)
    fp_of = {}
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in need:
                    rec = EntityRecord("X-" + p[0][2:], p[1], p[2])
                    fp_of[p[0]] = fingerprint_of(rec)
        if len(fp_of) >= len(need):
            break
    log(f"[{country}] fingerprints for {len(fp_of):,}/{len(need):,} candidates")

    # per-entity: group candidates into clusters, best prob per cluster
    eids_all = [r[0] for r in eval_recs]
    results = []
    t0 = time.time()
    pos0 = 0
    for i, ids in enumerate(cand_ids):
        p = probs[pos0:pos0 + len(ids)]
        pos0 += len(ids)
        best = {}
        for cid, pr in zip(ids, p):
            fp = fp_of.get(cid, cid)
            if pr > best.get(fp, (-1, None))[0]:
                best[fp] = (pr, cid, fp)
        clusters = sorted(best.values(), key=lambda x: -x[0])
        tr = eval_truth.get(eids_all[i], set())
        results.append((clusters, tr))
    log(f"[{country}] clustered in {time.time()-t0:.0f}s")

    # id -> fingerprint (for sibling propagation)
    # sweep two rule variants:
    #   A) cluster-best: predict only the best member per accepted cluster
    #   B) propagate:    predict ALL candidates sharing an accepted cluster's fingerprint
    # fingerprint -> all candidate ids per entity
    fp_members = {}
    for i, ids in enumerate(cand_ids):
        for cid in ids:
            fp = fp_of.get(cid, cid)
            fp_members.setdefault((i, fp), []).append(cid)

    best = {"A": (-1, None), "B": (-1, None)}
    for t in np.round(np.arange(0.65, 0.851, 0.01), 2):
        for K in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            def score(pred, tr):
                if not pred:
                    return 1.0 if not tr else 0.0
                tp = len(pred & tr)
                if tp == 0:
                    return 0.0
                pp = tp / len(pred)
                rr = tp / len(tr) if tr else 0.0
                return (1.25 * pp * rr) / (0.25 * pp + rr)

            accA = accB = 0.0
            for i, (clusters, tr) in enumerate(results):
                accepted = [(pr, cid, fp) for pr, cid, fp in clusters[:K] if pr >= t]
                predA = set(cid for _, cid, _ in accepted)
                predB = set()
                for _, _, fp in accepted:
                    predB.update(fp_members.get((i, fp), []))
                accA += score(predA, tr)
                accB += score(predB, tr)
            fA, fB = accA / len(results), accB / len(results)
            if fA > best["A"][0]:
                best["A"] = (fA, (t, K)); log(f"  new best {country} A: F0.5={fA:.4f} t={t} K={K}")
            if fB > best["B"][0]:
                best["B"] = (fB, (t, K)); log(f"  new best {country} B: F0.5={fB:.4f} t={t} K={K}")
    return best["A"][0], best["A"][1], best["B"][0], best["B"][1]


def main():
    ensure_hashseed()
    for c in ("us", "india"):
        feats = np.load(OUT / f"{c}_eval_feats.npy")
        with open(MODELS / "lgbm_model_v5.pkl", "rb") as f:
            clf = pickle.load(f)
        probs = clf.predict_proba(feats)[:, 1].astype(np.float32)
        del feats
        with open(OUT / f"{c}_eval_candids.pkl", "rb") as f:
            cand_ids = pickle.load(f)
        with open(OUT / f"{c}_eval_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        eval_truth = {k: set(v) for k, v in meta["eval_truth"].items()}
        log(f"---- {c}: cluster-level rule ----")
        # NOTE: needs sibling expansion of predictions to measure true value;
        # this first pass measures cluster-BEST-only acceptance.
        fA, cfgA, fB, cfgB = analyze(c, probs, cand_ids, meta["eval_recs"], eval_truth)
        log(f"[{c}] CLUSTER-BEST: F0.5={fA:.4f} cfg={cfgA}")
        log(f"[{c}] PROPAGATE:   F0.5={fB:.4f} cfg={cfgB}")


if __name__ == "__main__":
    main()
