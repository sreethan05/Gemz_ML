"""Phase C addon: evaluate (country, tag) x models from saved feats; update
calibration_report.json and rewrite {country}_best_rule.json if better.
Usage: python phase_c_addon.py <country> <tag>
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "scratch" / "v8"
MODELS = ROOT / "models"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_c import eval_grid, flatten  # noqa: E402


def main(country, tag):
    with open(SCR / f"{country}_candids_{tag}.pkl", "rb") as f:
        cand = pickle.load(f)
    with open(SCR / f"{country}_eval_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    truth = meta["eval_truth"]
    eids = [r[0] for r in meta["eval_recs"]]
    feats = np.load(SCR / f"{country}_feats_{tag}.npy", mmap_mode="r")
    eid_idx, tid, lab, sizes, flat, truth_sizes = flatten(cand, truth, eids)
    ths = np.round(np.arange(0.30, 0.92, 0.02), 3)
    mts = np.round(np.arange(0.30, 0.92, 0.02), 3)

    models = {}
    for m in ("m1", "m2"):
        p = MODELS / f"lgbm_model_v8_{m}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models[m] = pickle.load(f)
    if models:
        def ens(X):
            p1 = models["m1"].predict_proba(X)[:, 1]
            p2 = models["m2"].predict_proba(X)[:, 1]
            return np.column_stack([1 - (p1 + p2) / 2, (p1 + p2) / 2])
        models["ens"] = type("E", (), {"predict_proba": staticmethod(ens)})()

    report_path = SCR / "calibration_report.json"
    report = json.load(open(report_path)) if report_path.exists() else {}
    for mname, mdl in models.items():
        probs = mdl.predict_proba(feats)[:, 1].astype(np.float32)
        np.save(SCR / f"{country}_probs_{tag}_{mname}.npy", probs)
        res = eval_grid(eid_idx, tid, lab, sizes, truth_sizes, probs, ths, mts)
        best_f, bth, bmt = res[0]
        report[f"{country}|{tag}|{mname}"] = {
            "f05": round(best_f, 5), "th": bth, "mt": bmt,
            "pair_recall_cand": round(float(lab.mean()), 4),
            "top3": [(round(a, 5), b, d) for a, b, d in res[:3]],
        }
        print(f"{country}|{tag}|{mname}: F0.5={best_f:.5f} th={bth} mt={bmt}")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)

    # rewrite best rule if this tag/model beats the current one
    cur_path = SCR / f"{country}_best_rule.json"
    cur = json.load(open(cur_path)) if cur_path.exists() else None
    best = None
    for k, v in report.items():
        if k.startswith(f"{country}|") and (best is None or v["f05"] > best[0]):
            _, t, m = k.split("|")
            best = (v["f05"], t, m, v["th"], v["mt"])
    if best and (cur is None or best[0] > cur.get("f05", 0)):
        json.dump({"f05": best[0], "tag": best[1], "model": best[2],
                   "th": best[3], "mt": best[4]}, open(cur_path, "w"), indent=1)
        print(f"{country} best rule UPDATED -> tag={best[1]} model={best[2]} "
              f"F0.5={best[0]:.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
