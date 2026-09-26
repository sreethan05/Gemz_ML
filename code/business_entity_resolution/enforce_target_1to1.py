"""
Global Bipartite 1-to-1 Target Conflict Disambiguation
Amazon ML Challenge 2026 - Team Gemz

Ground Truth Invariant:
"Every S2 or S3 entity links to at most ONE S1 entity (0 multi-links)."

When multiple S1 entities claim the same S2 or S3 target entity, at most ONE can be true.
All other links are GUARANTEED false merges that destroy precision and F0.5.
This script resolves multi-claims by assigning each target entity uniquely to its best-matching S1 entity,
instantly eliminating hundreds of thousands of false positive merges!
"""

import sys
import os
import time
from pathlib import Path
from collections import defaultdict, Counter
import unicodedata
import re
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

jw = JaroWinkler.normalized_similarity
tsr = fuzz.token_sort_ratio

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def disambiguate_matching_results(matching_file: Path, output_file: Path, partitions_dir: Path):
    log("=" * 70)
    log("ENFORCING 1-TO-1 TARGET ASSIGNMENT (ZERO MULTI-LINKS)")
    log("=" * 70)

    # 1. First Pass: Read matching results and find duplicate target entities
    log(f"Reading {matching_file}...")
    s1_matches = {}
    target_counts = Counter()

    with open(matching_file, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                m_list = [x.strip() for x in parts[1].split(",") if x.strip()]
                s1_matches[s1_id] = m_list
                for tid in m_list:
                    target_counts[tid] += 1
            else:
                s1_matches[s1_id] = []

    total_s1 = len(s1_matches)
    total_preds = sum(len(v) for v in s1_matches.values())
    multi_targets = {tid for tid, count in target_counts.items() if count > 1}
    excess_links = sum(count - 1 for tid, count in target_counts.items() if count > 1)

    log(f"Total S1 entities: {total_s1:,}")
    log(f"Total predicted matches: {total_preds:,}")
    log(f"Unique target entities: {len(target_counts):,}")
    log(f"Target entities with MULTI-CLAIMS: {len(multi_targets):,} (generating {excess_links:,} guaranteed false merges!)")

    if not multi_targets:
        log("No multi-claims found! Output already satisfies strict 1-to-1 constraint.")
        return

    # 2. Map multi-targets to competing S1 entities
    target_to_s1 = defaultdict(list)
    for s1_id, m_list in s1_matches.items():
        for tid in m_list:
            if tid in multi_targets:
                target_to_s1[tid].append(s1_id)

    needed_s1 = {s1_id for s1_list in target_to_s1.values() for s1_id in s1_list}
    needed_oth = set(multi_targets)

    log(f"Loading record text for {len(needed_s1):,} S1 entities and {len(needed_oth):,} Target entities across partitions...")

    # Load needed S1 and Target texts from partitions
    s1_text = {}
    oth_text = {}

    for ctry in ["france", "us", "india"]:
        s1_path = partitions_dir / f"{ctry}_s1.tsv"
        oth_path = partitions_dir / f"{ctry}_other.tsv"

        if s1_path.exists():
            with open(s1_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if parts[0] in needed_s1 and len(parts) >= 3:
                        s1_text[parts[0]] = (clean_text(parts[1]), clean_text(parts[2]))

        if oth_path.exists():
            with open(oth_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if parts[0] in needed_oth and len(parts) >= 3:
                        oth_text[parts[0]] = (clean_text(parts[1]), clean_text(parts[2]))

    log("Resolving multi-claims via joint string similarity...")
    # 3. Disambiguate: for each multi-target, keep ONLY the single best S1 entity
    assigned_s1 = {}
    for tid, competitors in target_to_s1.items():
        if tid not in oth_text:
            assigned_s1[tid] = competitors[0]
            continue
        t_name, t_addr = oth_text[tid]
        best_s1 = None
        best_score = -1.0
        for s1_id in competitors:
            if s1_id not in s1_text:
                score = 0.0
            else:
                s_name, s_addr = s1_text[s1_id]
                score = jw(s_name, t_name) * 0.60 + (tsr(s_addr, t_addr) / 100.0) * 0.40
            if score > best_score:
                best_score = score
                best_s1 = s1_id
        assigned_s1[tid] = best_s1

    # 4. Filter S1 match lists so target entities are unique
    new_s1_matches = {}
    pruned_count = 0
    new_singletons = 0

    for s1_id, m_list in s1_matches.items():
        if not m_list:
            new_s1_matches[s1_id] = []
        else:
            kept = []
            for tid in m_list:
                if tid in multi_targets:
                    if assigned_s1.get(tid) == s1_id:
                        kept.append(tid)
                    else:
                        pruned_count += 1
                else:
                    kept.append(tid)
            new_s1_matches[s1_id] = kept
            if not kept and m_list:
                new_singletons += 1

    # 5. Write deduplicated matching results
    log(f"Writing 1-to-1 disambiguated results to {output_file}...")
    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, m_list in new_s1_matches.items():
            f.write(f"{s1_id}\t{','.join(sorted(m_list))}\n")

    final_preds = sum(len(v) for v in new_s1_matches.values())
    final_singletons = sum(1 for v in new_s1_matches.values() if not v)
    log("=" * 70)
    log(f"SUCCESS: Pruned {pruned_count:,} duplicate false merges!")
    log(f"New Singletons Cleanly Rescued: {new_singletons:,}")
    log(f"Final Total Matches: {final_preds:,} | Total Singletons: {final_singletons:,} ({final_singletons/total_s1*100:.2f}%)")
    log("=" * 70)

if __name__ == "__main__":
    m_in = Path("output/matching_results.tsv")
    m_out = Path("output/matching_results_1to1.tsv")
    parts = Path("dataset/test/partitions")
    disambiguate_matching_results(m_in, m_out, parts)
