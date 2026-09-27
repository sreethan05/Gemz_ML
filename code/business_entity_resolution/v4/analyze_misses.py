"""Analyze blocking misses: for eval entities whose true matches are absent
from the top-K candidates, classify why:
  A) no shared key at all with the true target (key-set gap)
  B) shared key exists but target ranked below cap (ranking gap)
Also prints example misses for eyeballing.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import keys_of, key_hashes

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "scratch" / "v4"


def load_truth_targets(country: str, need_ids: set) -> dict:
    """target_id -> (name, addr) for the eval truth targets of this country."""
    tgt = {}
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN / fn, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in need_ids:
                    tgt[p[0]] = (p[1], p[2])
        if len(tgt) >= len(need_ids):
            break
    return tgt


def analyze(country: str, cap: int = 40):
    with open(OUT / f"{country}_eval_candids.pkl", "rb") as f:
        cand_ids = pickle.load(f)
    with open(OUT / f"{country}_eval_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    from v4.core import EntityRecord
    eval_recs = {r[0]: EntityRecord(r[0], r[1], r[2]) for r in meta["eval_recs"]}
    eval_recs = [(r[0], EntityRecord(r[0], r[1], r[2])) for r in meta["eval_recs"]]
    truth = {k: set(v) for k, v in meta["eval_truth"].items()}

    need_t = set()
    for eid, tr in truth.items():
        need_t |= tr
    tgt_raw = load_truth_targets(country, need_t)
    tgt_rec = {tid: EntityRecord("X-" + tid[2:], nm, ad) for tid, (nm, ad) in tgt_raw.items()}

    n_pairs = hit = 0
    no_key = 0
    ranked_out = 0
    miss_examples = []
    for i, (eid, rec) in enumerate(eval_recs):
        tr = truth.get(eid, set())
        if not tr:
            continue
        have = set(cand_ids[i][:cap])
        for tid in tr:
            n_pairs += 1
            if tid in have:
                hit += 1
                continue
            t = tgt_rec.get(tid)
            if t is None:
                continue
            # Keep key types in the comparison.  Flattening to k[1] can report
            # a false shared key when, for example, a name token equals an
            # address token but the production index stores them separately.
            ks = set(keys_of(rec)) & set(keys_of(t))
            if ks:
                ranked_out += 1
                if len(miss_examples) < 12:
                    miss_examples.append(("RANK", eid, rec.name_norm[:38],
                                          tid, t.name_norm[:38], sorted(ks)[:3]))
            else:
                no_key += 1
                if len(miss_examples) < 12:
                    miss_examples.append(("NOKEY", eid, rec.name_norm[:38],
                                          tid, t.name_norm[:38], []))
    print(f"[{country}] true pairs={n_pairs:,} hit@{cap}={hit:,} "
          f"({hit/n_pairs:.4f}) | missed={n_pairs-hit:,}")
    print(f"  no-shared-key: {no_key:,} ({no_key/(n_pairs-hit):.1%} of misses)")
    print(f"  ranked-out-of-top{cap}: {ranked_out:,} ({ranked_out/(n_pairs-hit):.1%} of misses)")
    for tag, eid, nm, tid, tnm, ks in miss_examples:
        print(f"  [{tag}] {nm!r} <-> {tnm!r} keys={ks}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ("us", "india"):
        analyze(c, cap=40)
