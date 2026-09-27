"""v8 Phase C2: sibling-accept extension measured on holdout.

Rule: after the base (th, mt) accept, for each accepted pair (q, c*) also
accept co-candidates c' of q with fingerprint_of(c') == fingerprint_of(c*)
and prob(c') >= th_lo. Exact 1-to-1 sim applied after extension.

Measures: added pairs, added-pair precision, macro F0.5 delta.
Needs: phase_c's saved probs ({c}_probs_{tag}_{m}.npy) + calibration report.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCR8 = ROOT / "scratch" / "v8"
TRAIN = ROOT / "dataset" / "train"
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from v4.core import EntityRecord, fingerprint_of  # noqa: E402

COUNTRIES = ("us", "india")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def load_target_records(need):
    """target id -> EntityRecord for all needed candidate ids (one file pass)."""
    out = {}
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in need:
                    out[p[0]] = EntityRecord("X-" + p[0][2:], p[1], p[2])
        if len(out) >= len(need):
            break
    log(f"loaded {len(out):,} target records")
    return out


def f05_of(pred, tr):
    if not pred and not tr:
        return 1.0
    tp = len(pred & tr)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(tr)
    return (1.25 * p * r) / (0.25 * p + r)


def main(tag, model):
    with open(SCR8 / "calibration_report.json") as f:
        rep = json.load(f)
    need = set()
    arts = {}
    for c in COUNTRIES:
        with open(SCR8 / f"{c}_candids_{tag}.pkl", "rb") as f:
            cand = pickle.load(f)
        with open(SCR8 / f"{c}_eval_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        probs = np.load(SCR8 / f"{c}_probs_{tag}_{model}.npy")
        truth = {k: set(v) for k, v in meta["eval_truth"].items()}
        eids = [r[0] for r in meta["eval_recs"]]
        arts[c] = (cand, truth, eids, probs)
        for cs in cand:
            need.update(cs)
    tgt = load_target_records(need)
    fps = {tid: fingerprint_of(r) for tid, r in tgt.items()}

    for c in COUNTRIES:
        key = f"{c}|{tag}|{model}"
        if key not in rep:
            continue
        cand, truth, eids, probs = arts[c]
        # base best rule
        bth, bmt = rep[key]["th"], rep[key]["mt"]
        for th_lo in (0.30, 0.40, 0.50):
            # Reset the flattened-probability cursor for each independent
            # threshold-floor sweep.
            pos = 0
            f_sum = 0.0
            base_f_sum = 0.0
            added = added_good = 0
            base_preds = []
            extended_preds = []
            base_claims = {}
            extended_claims = {}
            for i, e in enumerate(eids):
                k = len(cand[i])
                pr = probs[pos:pos + k]
                cs = cand[i]
                pos += k
                tr = truth.get(e, set())
                base = {cs[j] for j in range(k)
                        if k and pr[j] >= bth and float(pr.max()) >= bmt}
                f_base = f05_of(base, tr)
                base_f_sum += f_base
                # sibling extension
                strong_fps = {fps.get(t, "") for t in base}
                strong_fps.discard("")
                extra = {cs[j] for j in range(k)
                         if cs[j] not in base and pr[j] >= th_lo
                         and fps.get(cs[j], "") in strong_fps}
                pred = base | extra
                added += len(extra)
                added_good += len(extra & tr)
                f_sum += f05_of(pred, tr)
                base_preds.append(base)
                extended_preds.append(pred)
                for predicted, claims in ((base, base_claims), (pred, extended_claims)):
                    for tid in predicted:
                        j = cs.index(tid)
                        claim = (float(pr[j]), e)
                        old = claims.get(tid)
                        if old is None or claim[0] > old[0] or (claim[0] == old[0] and e < old[1]):
                            claims[tid] = claim
            n = len(eids)
            base_1to1 = [{tid for tid in pred if base_claims.get(tid, (0.0, None))[1] == e}
                         for e, pred in zip(eids, base_preds)]
            ext_1to1 = [{tid for tid in pred if extended_claims.get(tid, (0.0, None))[1] == e}
                        for e, pred in zip(eids, extended_preds)]
            base_score = float(np.mean([f05_of(p, truth.get(e, set()))
                                        for e, p in zip(eids, base_1to1)]))
            ext_score = float(np.mean([f05_of(p, truth.get(e, set()))
                                       for e, p in zip(eids, ext_1to1)]))
            print(f"{c}|{tag}|{model} th={bth} mt={bmt} th_lo={th_lo:.2f}: "
                  f"post1to1 F0.5 {base_score:.5f} -> {ext_score:.5f} "
                  f"({(ext_score-base_score):+.5f})  added={added:,} "
                  f"added-precision={added_good/max(added,1):.2%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "n2",
         sys.argv[2] if len(sys.argv) > 2 else "ens")
