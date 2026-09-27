"""Build Gemz_submission_v8.zip per the challenge layout.

- output/matching_results.tsv + candidate_pairs.tsv  <- v8 merged files
- code/business_entity_resolution/ (src, README, requirements)
- Documentation_template.md
NOTE: writes a NEW zip; never touches the current board files.
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
V8 = ROOT / "scratch" / "v8"
STAGE = ROOT / "scratch" / "v8" / "stage"


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "output").mkdir(parents=True)
    code_dst = STAGE / "code" / "business_entity_resolution"
    code_dst.mkdir(parents=True)

    shutil.copy(OUT / "matching_results_v8.tsv", STAGE / "output" / "matching_results.tsv")
    shutil.copy(OUT / "candidate_pairs_v8.tsv", STAGE / "output" / "candidate_pairs.tsv")
    shutil.copy(ROOT / "Documentation_template.md", STAGE / "Documentation_template.md")

    src = ROOT / "code" / "business_entity_resolution"
    for name in ("README.md", "requirements.txt"):
        shutil.copy(src / name, code_dst / name)
    sdst = code_dst / "src"
    sdst.mkdir()
    for p in sorted(src.glob("*.py")):
        shutil.copy(p, sdst / p.name)
    # v8 scripts (retrieval/feats/phase_c/run_test/orchestrate/finish)
    v8dst = code_dst / "src" / "v8"
    v8dst.mkdir()
    for name in ("retrieval_v8.py", "feats8.py", "phase_c.py", "run_test_v8.py",
                 "orchestrate.py", "finish_v8.py"):
        shutil.copy(V8 / name, v8dst / name)

    zpath = ROOT / "Gemz_submission_v8.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE))
    print(f"wrote {zpath} ({zpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
