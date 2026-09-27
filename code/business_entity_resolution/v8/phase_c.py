"""v8 Phase C: train selection models + calibrate decision rules on holdout.

Fully vectorized: each eval artifact is flattened to (eid_idx, tid_dense, lab,
sizes); a (threshold, min_top) grid including an exact 1-to-1 disambiguation
simulation is evaluated with bincount/reduceat — no per-combo python loops.

Inputs (scratch/v8/): {c}_feats_{tag}.npy, {c}_candids_{tag}.pkl,
{c}_eval_meta.pkl, {c}_train_X.npy, {c}_train_y.npy.
Outputs: models/lgbm_model_v8_*.pkl, scratch/v8/calibration_report.json,
{c}_best_rule.json (production decision rules), {c}_eval_probs_{tag}_{m}.npy.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "scratch" / "v8"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))

COUNTRIES = ("us", "india")
TAGS = ("v4base", "n1", "n2", "n3")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def train_models():
    Xs, ys = [], []
    for c in COUNTRIES:
        Xs.append(np.load(SCR / f"{c}_train_X.npy"))
        ys.append(np.load(SCR / f"{c}_train_y.npy"))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    log(f"train matrix: {X.shape[0]:,} pairs, {int(y.sum()):,} pos ({y.mean():.2%})")
    import lightgbm as lgb
    m1 = lgb.LGBMClassifier(
        n_estimators=900, learning_rate=0.045, num_leaves=127, max_depth=11,
        min_child_samples=25, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=17, n_jobs=2)
    m2 = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.06, num_leaves=95, max_depth=9,
        min_child_samples=35, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, extra_trees=True, random_state=123, n_jobs=2)
    t0 = time.time()
    m1.fit(X, y)
    log(f"M1 trained ({time.time()-t0:.0f}s)")
    t0 = time.time()
    m2.fit(X, y)
    log(f"M2 trained ({time.time()-t0:.0f}s)")
    with open(MODELS / "lgbm_model_v8_m1.pkl", "wb") as f:
        pickle.dump(m1, f)
    with open(MODELS / "lgbm_model_v8_m2.pkl", "wb") as f:
        pickle.dump(m2, f)
    return m1, m2


def flatten(cand, truth, eids):
    """-> eid_idx, tid_dense, lab, sizes, tid_list (dense->id), truth_sizes."""
    sizes = np.array([len(c) for c in cand], dtype=np.int64)
    flat = [t for cs in cand for t in cs]
    dense = {}
    tid = np.empty(len(flat), dtype=np.int64)
    for i, t in enumerate(flat):
        v = dense.get(t)
        if v is None:
            v = len(dense)
            dense[t] = v
        tid[i] = v
    eid_idx = np.repeat(np.arange(len(cand)), sizes)
    lab = np.zeros(len(flat), dtype=bool)
    pos = 0
    truth_sizes = np.zeros(len(cand), dtype=np.int64)
    for i, e in enumerate(eids):
        tr = truth.get(e, ())
        truth_sizes[i] = len(tr)
        trset = set(tr)
        for k, t in enumerate(cand[i]):
            lab[pos + k] = t in trset
        pos += len(cand[i])
    return eid_idx, tid, lab, sizes, flat, truth_sizes


def f05_vec(tp, pred_n, truth_n):
    tp = tp.astype(np.float64)
    pred_n = pred_n.astype(np.float64)
    truth_n = truth_n.astype(np.float64)
    f = np.zeros(len(tp), dtype=np.float64)
    both = (pred_n > 0) & (truth_n > 0)
    p = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    r = np.divide(tp, truth_n, out=np.zeros_like(tp), where=truth_n > 0)
    denom = 0.25 * p + r
    ok = both & (denom > 0) & (tp > 0)
    f[ok] = 1.25 * p[ok] * r[ok] / denom[ok]
    f[(pred_n == 0) & (truth_n == 0)] = 1.0
    return f


def eval_grid(eid_idx, tid, lab, sizes, truth_sizes, probs, ths, mts, eid_tiebreak):
    """Vectorized grid incl. exact 1-to-1 prune simulation. Returns best combos.

    mt semantics match v4 production: min_top applies to the entity's max prob
    over ALL its candidates (th-independent), computed once via reduceat.
    """
    n_ent = len(sizes)
    starts = np.flatnonzero(np.r_[True, eid_idx[1:] != eid_idx[:-1]])
    ent_max = np.full(n_ent, -np.inf, dtype=np.float32)
    if len(starts):
        ent_max[np.unique(eid_idx[starts])] = np.maximum.reduceat(probs, starts)
    ent_max_ok = ent_max  # indexed via eid_idx per pair
    results = []
    for th in ths:
        keep = probs >= th
        keep_base = keep & (ent_max_ok[eid_idx] > -np.inf)
        for mt in mts:
            keep2 = keep_base & (ent_max_ok[eid_idx] >= mt)
            # exact 1-to-1: per target, keep only the highest-prob claim
            idx = np.flatnonzero(keep2)
            if len(idx):
                pr = probs[idx]
                tids = tid[idx]
                s1 = eid_idx[idx]
                # Match production's deterministic tie-break: lexicographic S1
                # ID after probability, rather than holdout row order.
                order = np.lexsort((eid_tiebreak[s1], -pr, tids))
                t_s = tids[order]
                first = np.r_[True, t_s[1:] != t_s[:-1]]
                win_pos = order[first]
                keep3 = np.zeros(len(probs), dtype=bool)
                keep3[idx[win_pos]] = True
            else:
                keep3 = keep2
            tp = np.bincount(eid_idx[keep3 & lab], minlength=n_ent)
            pred_n = np.bincount(eid_idx[keep3], minlength=n_ent)
            f = f05_vec(tp, pred_n, truth_sizes)
            results.append((float(f.mean()), th, mt))
    results.sort(reverse=True)
    return results


def main():
    # 1) train
    m1, m2 = train_models()

    class Ens:
        @staticmethod
        def predict_proba(X):
            p = (m1.predict_proba(X)[:, 1] + m2.predict_proba(X)[:, 1]) / 2
            return np.column_stack([1 - p, p])

    models = {"m1": m1, "m2": m2, "ens": Ens()}
    report = {}
    ths = np.round(np.arange(0.30, 0.92, 0.02), 3)
    mts = np.round(np.arange(0.30, 0.92, 0.02), 3)

    for c in COUNTRIES:
        for tag in TAGS:
            fpath = SCR / f"{c}_feats_{tag}.npy"
            if not fpath.exists():
                log(f"skip {c}/{tag} (no feats)")
                continue
            with open(SCR / f"{c}_candids_{tag}.pkl", "rb") as f:
                cand = pickle.load(f)
            with open(SCR / f"{c}_eval_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            truth = {k: v for k, v in meta["eval_truth"].items()}
            eids = [r[0] for r in meta["eval_recs"]]
            eid_tiebreak = np.empty(len(eids), dtype=np.int32)
            eid_tiebreak[np.argsort(np.asarray(eids, dtype=object), kind="stable")] = \
                np.arange(len(eids), dtype=np.int32)
            feats = np.load(fpath, mmap_mode="r")
            eid_idx, tid, lab, sizes, flat, truth_sizes = flatten(cand, truth, eids)
            pair_recall = lab.mean() if lab.size else 0.0
            for mname, mdl in models.items():
                probs = mdl.predict_proba(feats)[:, 1].astype(np.float32)
                np.save(SCR / f"{c}_probs_{tag}_{mname}.npy", probs)
                res = eval_grid(eid_idx, tid, lab, sizes, truth_sizes, probs,
                                ths, mts, eid_tiebreak)
                best_f, bth, bmt = res[0]
                report[f"{c}|{tag}|{mname}"] = {
                    "f05": round(best_f, 5), "th": bth, "mt": bmt,
                    "pair_recall_cand": round(float(lab.mean()), 4),
                    "top3": [(round(a, 5), b, d) for a, b, d in res[:3]],
                }
                log(f"{c}/{tag}/{mname}: F0.5={best_f:.5f} @ th={bth} mt={bmt} "
                    f"(cand pair-recall {pair_recall:.4f})")
    with open(SCR / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=1)
    log("saved calibration_report.json")


if __name__ == "__main__":
    main()
