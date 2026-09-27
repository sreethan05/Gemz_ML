"""v4 train + calibrate: LightGBM on production-scale pairs, then per-country
(cap, threshold, min_top, rescue) calibration on the honest held-out eval set.

Calibration counts are O(1) per entity per config via prefix sums over
prob-sorted candidates, so the full grid sweeps in seconds.
"""
import gc
import json
import pickle
import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import ensure_hashseed, entity_f05

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "scratch" / "v4"
MODELS = ROOT / "models"

CAPS = (10, 12, 15, 20, 25, 30, 40)
TS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
RESCUES = ("off", "gated", "full")
# feature indices used by the rescue rule
I_NSORT, I_ASORT, I_PINMATCH, I_NUMOV = 2, 11, 20, 17

COUNTRIES = ("us", "india")
TEST_SHARE = {"us": 663106 / 1732544, "india": 809986 / 1732544, "france": 259452 / 1732544}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def f05_from_counts(cnt: int, tp: int, tot: int) -> float:
    if cnt == 0:
        return 1.0 if tot == 0 else 0.0
    if tp == 0:
        return 0.0
    p = tp / cnt
    r = tp / tot
    return (1.25 * p * r) / (0.25 * p + r)


def train():
    Xs, ys = [], []
    for c in COUNTRIES:
        Xs.append(np.load(OUT / f"{c}_train_X.npy"))
        ys.append(np.load(OUT / f"{c}_train_y.npy"))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    log(f"Training on {len(X):,} pairs ({int(y.sum()):,} positive)...")

    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    if variant == "deep":
        clf = lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.045, num_leaves=96, max_depth=10,
            min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
            random_state=42, n_jobs=4,
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.06, num_leaves=64, max_depth=8,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=4,
        )
    clf.fit(X, y)
    MODELS.mkdir(exist_ok=True)
    if variant == "ensemble":
        model_name = "lgbm_model_v6_ensemble.pkl"
    else:
        model_name = "lgbm_model_v4.pkl" if variant == "base" else "lgbm_model_v5.pkl"
    with open(MODELS / model_name, "wb") as f:
        pickle.dump(clf, f)
    log(f"model saved -> models/{model_name}")

    if variant == "ensemble":
        import lightgbm as _lgb
        from v4.core import EnsembleModel
        clfs = [clf]
        for seed, nl, mc in ((7, 80, 25), (13, 48, 60)):
            m2 = _lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.05, num_leaves=nl, max_depth=-1,
                min_child_samples=mc, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=4,
            )
            m2.fit(X, y)
            clfs.append(m2)
            log(f"ensemble member {seed} fitted")
        ens = EnsembleModel(clfs)
        with open(MODELS / "lgbm_model_v6_ensemble.pkl", "wb") as f:
            pickle.dump(ens, f)
        log("ensemble saved -> models/lgbm_model_v6_ensemble.pkl")
        return ens, model_name, variant
    return clf, model_name, variant


def main():
    ensure_hashseed()
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    if variant == "sweeponly":
        model_name = sys.argv[2]
        with open(MODELS / model_name, "rb") as f:
            clf = pickle.load(f)
        log(f"sweep-only: loaded {model_name}")
    else:
        clf, model_name, variant = train()

    per_country = {}
    for c in COUNTRIES:
        log(f"---- calibration: {c} ----")
        feats = np.load(OUT / f"{c}_eval_feats.npy")
        if isinstance(clf, list):
            probs = np.mean([m.predict_proba(feats)[:, 1] for m in clf], axis=0).astype(np.float32)
        else:
            probs = clf.predict_proba(feats)[:, 1].astype(np.float32)
        elig_gated = ((feats[:, I_ASORT] >= 0.90) &
                      ((feats[:, I_PINMATCH] > 0.5) | (feats[:, I_NUMOV] >= 0.5)) &
                      (feats[:, I_NSORT] >= 0.85))
        elig_full = ((feats[:, I_ASORT] >= 0.90) &
                     ((feats[:, I_PINMATCH] > 0.5) | (feats[:, I_NUMOV] >= 0.5)))
        del feats
        gc.collect()

        s1_idx = np.load(OUT / f"{c}_eval_s1idx.npy")
        with open(OUT / f"{c}_eval_candids.pkl", "rb") as f:
            cand_ids = pickle.load(f)
        with open(OUT / f"{c}_eval_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        eval_truth = {k: set(v) for k, v in meta["eval_truth"].items()}
        eids = [r[0] for r in meta["eval_recs"]]
        n_ent = len(cand_ids)

        # contiguous row ranges per entity
        row_ranges = []
        start = 0
        for ids in cand_ids:
            row_ranges.append((start, start + len(ids)))
            start += len(ids)

        best = None
        # per-entity prob-sorted python structures over ALL candidates;
        # cap is applied at sweep time as "top-cap by model probability"
        ents = []
        for i in range(n_ent):
            s0, s1_ = row_ranges[i]
            k = s1_ - s0
            tr = eval_truth.get(eids[i], ())
            p = probs[s0:s0 + k]
            order = np.argsort(-p, kind="stable")
            o = order.tolist()
            p_desc = p[o].tolist()
            g = elig_gated[s0:s0 + k][o].tolist()
            fl = elig_full[s0:s0 + k][o].tolist()
            t_desc = [1 if cand_ids[i][m] in tr else 0 for m in o]
            T = [0] * (k + 1)
            G = [0] * (k + 1)
            F = [0] * (k + 1)
            TG = [0] * (k + 1)
            TF = [0] * (k + 1)
            for j in range(k):
                T[j + 1] = T[j] + t_desc[j]
                G[j + 1] = G[j] + (1 if g[j] else 0)
                F[j + 1] = F[j] + (1 if fl[j] else 0)
                TG[j + 1] = TG[j] + (t_desc[j] and g[j])
                TF[j + 1] = TF[j] + (t_desc[j] and fl[j])
            ents.append((p_desc, T, G, F, TG, TF, len(tr)))
        log(f"  structures over all candidates: {time.strftime('%H:%M:%S')}")

        best = None
        for cap in CAPS:
            t0 = time.time()
            for t in TS:
                # prefix sums (python lists)
                T = [0] * (k + 1)
                G = [0] * (k + 1)
                F = [0] * (k + 1)
                TG = [0] * (k + 1)
                TF = [0] * (k + 1)
                for j in range(k):
                    T[j + 1] = T[j] + t_desc[j]
                    G[j + 1] = G[j] + (1 if g[j] else 0)
                    F[j + 1] = F[j] + (1 if fl[j] else 0)
                    TG[j + 1] = TG[j] + (t_desc[j] and g[j])
                    TF[j + 1] = TF[j] + (t_desc[j] and fl[j])
                tot = len(tr)
                ents.append((p_desc, T, G, F, TG, TF, tot))
            log(f"  cap={cap}: structures in {time.time()-t0:.1f}s")

            t0 = time.time()
            for t in TS:
                for mt in tuple(m for m in TS if m >= t):
                    for resc in RESCUES:
                        acc = 0.0
                        for (p_desc, T, G, F, TG, TF, tot) in ents:
                            n = len(p_desc)
                            if n == 0:
                                acc += 1.0 if tot == 0 else 0.0
                                continue
                            L = min(cap, n)
                            asc = p_desc[:L][::-1]
                            k_ge = L - bisect_left(asc, t)   # p>=t within top-L
                            if resc == "off":
                                cnt = k_ge
                                tp = T[k_ge]
                                best_eff = p_desc[0]
                            else:
                                E = G if resc == "gated" else F
                                ET = TG if resc == "gated" else TF
                                cnt = k_ge + (E[L] - E[k_ge])   # eligible inside top-L
                                tp = T[k_ge] + (ET[L] - ET[k_ge])
                                best_eff = p_desc[0]
                                if E[L] > 0 and t > best_eff:
                                    best_eff = t
                            if best_eff < mt:
                                cnt = 0
                            if cnt == 0:
                                acc += 1.0 if tot == 0 else 0.0
                            else:
                                tp = min(tp, cnt, tot)
                                if tp == 0:
                                    continue  # f05 = 0
                                p_ = tp / cnt
                                r_ = tp / tot
                                acc += (1.25 * p_ * r_) / (0.25 * p_ + r_)
                        f = acc / n_ent
                        if best is None or f > best[0]:
                            best = (f, cap, t, mt, resc)
                            log(f"  new best {c}: F0.5={f:.4f} cap={cap} t={t:.2f} "
                                f"mt={mt:.2f} rescue={resc}")
            log(f"  cap={cap}: swept in {time.time()-t0:.0f}s")

        f_best, cap_b, t_b, mt_b, resc_b = best
        # blocking recall at the winning cap
        hit = tot_pairs = 0
        for i in range(n_ent):
            tr = eval_truth.get(eids[i], ())
            tot_pairs += len(tr)
            hit += sum(1 for cid in cand_ids[i][:cap_b] if cid in tr)
        log(f"[{c}] BEST: F0.5={f_best:.4f} cap={cap_b} t={t_b:.2f} mt={mt_b:.2f} "
            f"rescue={resc_b} | blocking recall@cap={hit:,}/{tot_pairs:,} "
            f"({hit/max(tot_pairs,1):.4f})")
        per_country[c] = {"f05": f_best, "cap": cap_b, "threshold": t_b,
                          "min_top": mt_b, "rescue": resc_b,
                          "blocking_recall_at_cap": hit / max(tot_pairs, 1),
                          "n_eval_entities": n_ent}
        del probs, ents
        gc.collect()

    w = TEST_SHARE
    comb = (per_country["us"]["f05"] * w["us"] + per_country["india"]["f05"] * w["india"]
            + per_country["india"]["f05"] * w["france"])
    log(f"COMBINED expected macro-F0.5 (france uses india config): {comb:.4f}")

    cfg = {
        "countries": {c: per_country[c] for c in COUNTRIES},
        "france_uses": "india",
        "combined_expected_f05": comb,
        "model": f"models/{model_name}",
        "variant": variant,
        "cap_semantics": "model-ranked",
    }
    if variant == "sweeponly":
        cfg_name = sys.argv[3] if len(sys.argv) > 3 else "v6_config.json"
    else:
        cfg_name = {"base": "v4_config.json", "deep": "v5_config.json",
                    "ensemble": "v6_config.json"}[variant]
    with open(MODELS / cfg_name, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log(f"config saved -> models/{cfg_name}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("FATAL: out of memory. Close other applications and re-run.", flush=True)
        sys.exit(3)
