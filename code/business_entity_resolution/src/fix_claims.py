"""Regenerate claims for india shard-1 sub-shards (s1a-d) on box A.
Recomputes model probs for each accepted pair (deterministic), writes
claims_v8_india_s1{a,b,c,d}.tsv next to the mr files."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
sys.path.insert(0, str(ROOT / "scratch" / "v8"))
sys.path.insert(0, str(ROOT / "scratch" / "antigravity"))

from v4.core import EntityRecord, ensure_hashseed  # noqa: E402
from cloud_shard import get_pool_csrs, load_model, w_process  # noqa: E402


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def main():
    ensure_hashseed()
    with open(ROOT / "scratch" / "v8" / "india_best_rule.json") as f:
        rule = json.load(f)
    rcfg = json.load(open(ROOT / "scratch" / "v8" / "variant_params.json"))[rule["tag"]]
    clf = load_model(rule)
    pool, csrs = get_pool_csrs("india")

    # id -> pool index
    id2idx = {}
    for j in range(pool.n):
        id2idx[pool.get_id(j)] = j

    # S1 raw records for the shard-1 range, in stream order
    lo, hi = 202497, 404994
    all_lines = []
    with open(ROOT / "dataset" / "test" / "test_source1.tsv", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3].strip().lower() == "india":
                all_lines.append(line)
            if len(all_lines) >= hi:
                break
    lines = all_lines[lo:hi]
    del all_lines
    log(f"{len(lines):,} s1 entities in range")

    for tag in ("a", "b", "c", "d"):
        m_path = ROOT / "output" / f"mr_v8_india_s1{tag}.tsv"
        claims_out = ROOT / "scratch" / "v8" / f"claims_v8_india_s1{tag}.tsv"
        if claims_out.exists():
            log(f"claims s1{tag} already present")
            continue
        # entities of this sub-shard: mr rows with any match
        matched = {}
        for line in open(m_path, encoding="utf-8"):
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest:
                matched[s1] = rest.split(",")
        log(f"s1{tag}: {len(matched):,} entities with matches")
        # raw records for those entities
        raw = {}
        with open(ROOT / "dataset" / "test" / "test_source1.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in matched:
                    raw[p[0]] = line
        out = []
        from feats8 import batch_pair_features
        import numpy as np
        for s1, rawline in raw.items():
            p = rawline.rstrip("\r\n").split("\t")
            r = EntityRecord(p[0], p[1], p[2])
            cids = matched[s1]
            idxs = [id2idx[c] for c in cids if c in id2idx]
            if not idxs:
                continue
            cands = [pool.record(j) for j in idxs]
            probs = clf.predict_proba(batch_pair_features(r, cands))[:, 1]
            for cid, pr in zip(cids, probs):
                out.append(f"{s1}\t{cid}\t{pr:.6f}\n")
        with open(claims_out, "w", encoding="utf-8") as g:
            g.writelines(out)
        log(f"s1{tag}: wrote {len(out):,} claims")


if __name__ == "__main__":
    main()
