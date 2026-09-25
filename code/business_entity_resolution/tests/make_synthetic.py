#!/usr/bin/env python3
"""Generate synthetic challenge-shaped data for local end-to-end testing.

Reproduces the documented noise patterns: name typos / abbreviation /
word-order transposition, legal-suffix variation, address abbreviation,
missing components, landmark references, transliterations, US/India train
countries and an unseen France test country, singletons and multi-matches.

    python3 tests/make_synthetic.py --out tests/dataset --seed 7
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

BRANDS = ["Sunrise", "Krishna", "Global", "Vardhman", "Lakshmi", "Blue Orchid",
          "Shree Ganesh", "Annapurna", "Evergreen", "Metro", "Silver Line",
          "Deccan", "Ashwin", "Nova", "Royal", "Premier", "Kaveri", "Nandini",
          "Orbit", "Zenith", "Tulsi", "Sagar", "Pearl", "Crestline", "Aurora"]
TYPES = ["Bakery", "Electronics", "Textiles", "Pharma", "Logistics", "Cafe",
         "Motors", "Steel", "Enterprises", "Provisions", "Hardware", "Traders"]
SUFFIXES = ["Pvt Ltd", "Private Limited", "Ltd", "Limited", "LLC", "Inc",
            "Corp", "Corporation", "LLP", "Co"]

US_STREETS = ["Maple", "Cedar", "Oakwood", "Riverside", "Elm", "Highland",
              "Bayview", "Sunset", "Willow", "Juniper"]
US_STYPES = ["Road", "Street", "Avenue", "Boulevard", "Lane", "Drive"]
US_CITIES = ["Springfield", "Riverton", "Fairview", "Greenville", "Madison",
             "Georgetown", "Clinton", "Salem", "Bristol", "Dover"]
IN_STREETS = ["MG", "Station", "Temple", "Gandhi", "Nehru", "Main", "Park",
              "Ring", "Lake", "Market"]
IN_AREAS = ["Green Park", "Shastri Nagar", "Indira Colony", "Vidya Vihar",
            "Ranjini Layout", "Saraswathipuram", "Balaji Nagar", "Ashok Vihar"]
IN_CITIES = ["Pune", "Mysuru", "Nagpur", "Vijayawada", "Coimbatore", "Patna",
             "Indore", "Rajkot", "Solan", "Warangal"]
FR_STREETS = ["Victor Hugo", "Jean Jaures", "Republique", "Gare", "Liberte",
              "Pasteur", "Moliere", "Carnot"]
FR_CITIES = ["Lyon", "Nantes", "Toulouse", "Reims", "Angers", "Dijon"]

TRANSLIT = {"Krishna": "Krushna", "Lakshmi": "Laxmi", "Shree": "Sri",
            "Annapurna": "Annapoorna", "Sagar": "Saagar", "Tulsi": "Thulasi"}

LANDMARKS = ["Near SBI ATM", "Opp. Bus Stand", "Near Metro Station",
             "Behind Old Market", "Near HDFC Bank", "Opp. Water Tank"]


def typo(rng, s):
    if len(s) < 4 or rng.random() > 0.35:
        return s
    i = rng.randrange(1, len(s) - 2)
    op = rng.random()
    if op < 0.4:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]          # transpose
    if op < 0.7:
        return s[:i] + rng.choice("aeiou") + s[i:]           # insertion
    return s[:i] + rng.choice("aeioustn") + s[i + 1:]        # substitution


def noisy_name(rng, base):
    name = base
    if rng.random() < 0.4:  # legal suffix variation
        for a, b in (("Private Limited", "Pvt Ltd"), ("Corporation", "Corp"),
                     ("Limited", "Ltd"), ("Inc", "LLC"), (" and ", " & ")):
            name = name.replace(a, b) if a in name and rng.random() < 0.7 else name
    if rng.random() < 0.25:
        words = name.split()
        if len(words) > 2:
            rng.shuffle(words)
            name = " ".join(words)
    if rng.random() < 0.3:
        name = typo(rng, name)
    for k, v in TRANSLIT.items():                            # transliteration
        if k in name and rng.random() < 0.5:
            name = name.replace(k, v)
    return name


def noisy_addr(rng, addr):
    if rng.random() < 0.5:  # abbreviations
        for a, b in ((" Road", " Rd"), (" Street", " St"), (" Avenue", " Ave"),
                    (" Marg", " Mgr"), (" Boulevard", " Blvd"), (" Lane", " Ln"),
                    (" Opposite", " Opp"), (" Near", " Nr")):
            addr = addr.replace(a, b) if rng.random() < 0.6 else addr
    if rng.random() < 0.3:  # drop a component
        parts = [p for p in addr.split(",") if p.strip()]
        if len(parts) > 2:
            parts.pop(rng.randrange(1, len(parts) - 1))
            addr = ", ".join(parts)
    if rng.random() < 0.2:  # landmark reference
        addr = addr + ", " + rng.choice(LANDMARKS)
    if rng.random() < 0.25:
        addr = typo(rng, addr)
    if rng.random() < 0.2:  # component reorder
        parts = [p for p in addr.split(",") if p.strip()]
        if len(parts) > 2:
            i, j = sorted(rng.sample(range(len(parts)), 2))
            parts[i], parts[j] = parts[j], parts[i]
            addr = ", ".join(parts)
    return addr


def make_business(rng, country):
    brand = rng.choice(BRANDS)
    btype = rng.choice(TYPES)
    suffix = rng.choice(SUFFIXES)
    name = f"{brand} {btype} {suffix}"
    if country == "US":
        addr = (f"{rng.randint(10, 9999)} {rng.choice(US_STREETS)} "
                f"{rng.choice(US_STYPES)}, {rng.choice(US_CITIES)}, "
                f"{rng.choice(['CA', 'TX', 'NY', 'IL', 'WA'])} "
                f"{rng.randint(10000, 99500)}")
    elif country == "India":
        addr = (f"No. {rng.randint(1, 250)}, {rng.choice(IN_STREETS)} Marg, "
                f"{rng.choice(IN_AREAS)}, {rng.choice(IN_CITIES)} - "
                f"{rng.randint(110001, 799999)}, "
                f"{rng.choice(['Maharashtra', 'Karnataka', 'Gujarat', 'Bihar'])}")
    else:  # France - unseen in training
        addr = (f"{rng.randint(1, 180)} {rng.choice(FR_STREETS)}, "
                f"{rng.randint(1000, 9500) * 10} {rng.choice(FR_CITIES)}")
    return {"country": country, "name": name, "addr": addr}


def confusable(rng, biz):
    """A DIFFERENT business with a similar name but a different city
    (hard negative: must not be merged)."""
    other = make_business(rng, biz["country"])
    words = biz["name"].split()
    keep = " ".join(words[:2]) if len(words) > 2 else words[0]
    other["name"] = keep + " " + rng.choice(SUFFIXES)
    return other


def write_tsv(path: Path, rows, cols):
    """rows: iterable of tuples/str aligned with `cols`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def gen_split(rng, out: Path, tag, n_s1, n_s2, n_s3, countries, with_truth):
    pool = [make_business(rng, rng.choice(countries)) for _ in range(n_s1 + n_s2 + n_s3 + 400)]
    used = 0

    s1_rows, other_rows, truth = [], [], {}
    idc = {"S2": 0, "S3": 0}

    def add_other(biz):
        src = rng.choice(["S2", "S3"])
        idc[src] += 1
        other_rows.append((f"{src}-{idc[src]:05d}", noisy_name(rng, biz["name"]),
                           noisy_addr(rng, biz["addr"]), biz["country"]))
        return other_rows[-1][0]

    for k in range(n_s1):
        biz = pool[used]; used += 1
        eid = f"S1-{k + 1:05d}"
        s1_rows.append((eid, noisy_name(rng, biz["name"]),
                        noisy_addr(rng, biz["addr"]), biz["country"]))
        matches = []
        if rng.random() > 0.35:  # ~65% have 1..3 matches
            for _ in range(rng.choice([1, 1, 2, 2, 3])):
                matches.append(add_other(biz))
        truth[eid] = matches
        if rng.random() < 0.12:  # hard negative: confusable sibling
            add_other(confusable(rng, biz))

    while len([r for r in other_rows if r[0].startswith("S2-")]) < n_s2:
        biz = pool[used]; used += 1
        add_other(biz)
    while len([r for r in other_rows if r[0].startswith("S3-")]) < n_s3:
        biz = pool[used]; used += 1
        add_other(biz)
    rng.shuffle(other_rows)

    cols = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(out / f"{tag}_source1.tsv", s1_rows, cols)
    write_tsv(out / f"{tag}_source2.tsv",
              [r for r in other_rows if r[0].startswith("S2-")], cols)
    write_tsv(out / f"{tag}_source3.tsv",
              [r for r in other_rows if r[0].startswith("S3-")], cols)
    if with_truth:
        write_tsv(out / f"{tag}_ground_truth.tsv",
                  [(e, ",".join(m)) for e, m in truth.items()],
                  ["source1_entity_id", "matched_entity_ids"])
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/dataset")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.out)

    train_truth = gen_split(rng, root / "train", "train",
                            n_s1=1200, n_s2=1400, n_s3=1100,
                            countries=["US", "US", "India", "India"],
                            with_truth=True)
    test_truth = gen_split(rng, root / "test", "test",
                           n_s1=800, n_s2=950, n_s3=800,
                           countries=["US", "US", "India", "India", "France", "France"],
                           with_truth=False)

    # keep a hidden test truth copy for local scoring only (NOT part of the
    # challenge's shipped files)
    write_tsv(root / "test_hidden_truth.tsv",
              [(e, ",".join(m)) for e, m in test_truth.items()],
              ["source1_entity_id", "matched_entity_ids"])
    n_single = sum(1 for m in train_truth.values() if not m)
    print(f"train: {len(train_truth)} S1 entities ({n_single} singletons)")
    print(f"test : {len(test_truth)} S1 entities "
          f"({sum(1 for m in test_truth.values() if not m)} singletons)")


if __name__ == "__main__":
    main()
