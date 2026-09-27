"""Chain: wait for v4 finisher -> archive v4 artifacts -> run v5 prep+train+calibrate.

v5 = domain-concat keys + 4 chain-discrimination features + deeper model.
Stops after calibration so the v4-vs-v5 config decision is made by measurement.
"""
import shutil
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / "scratch" / "v4"
V4 = ROOT / "code" / "business_entity_resolution" / "v4"
MODELS = ROOT / "models"
OUTD = ROOT / "output"
ZIP = ROOT / "Gemz_submission.zip"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    lf = open(SCR / "chain_sub.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "1"
    r = subprocess.run([sys.executable, *cmd], stdout=lf, stderr=subprocess.STDOUT,
                       env=env)
    lf.close()
    tail = (SCR / "chain_sub.log").read_text(encoding="utf-8").strip().splitlines()[-8:]
    print("\n".join(tail), flush=True)
    return r.returncode


log("waiting for v4 zip (finisher)...")
t0 = time.time()
while not ZIP.exists():
    if time.time() - t0 > 4 * 3600:
        log("TIMEOUT waiting for v4 zip")
        sys.exit(2)
    time.sleep(60)
log("v4 zip present")

# archive v4 artifacts so v5 starts clean
bak = SCR / "v4_artifacts_backup"
bak.mkdir(exist_ok=True)
for p in SCR.glob("*_train_*.npy"):
    shutil.move(str(p), bak / p.name)
for p in SCR.glob("*_eval_*.pkl"):
    shutil.move(str(p), bak / p.name)
for p in SCR.glob("*_eval_*.npy"):
    shutil.move(str(p), bak / p.name)
for c in ("france", "us", "india"):
    for stem in (f"matching_results_{c}.tsv", f"candidate_pairs_{c}.tsv",
                 f"claims_{c}.tsv"):
        src = SCR / stem if stem.startswith("claims") else OUTD / stem
        if src.exists():
            shutil.move(str(src), bak / stem)
log("v4 artifacts archived")

if run(["code/business_entity_resolution/v4/prep.py"]) != 0:
    sys.exit(3)
if run(["code/business_entity_resolution/v4/train_calibrate.py", "deep"]) != 0:
    sys.exit(4)
log("V5 PREP+TRAIN+CALIBRATE COMPLETE - awaiting config comparison")
