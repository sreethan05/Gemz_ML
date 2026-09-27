"""Automated Orchestrator for v5 Pipeline.

Monitors training completion, runs parallel test inference, merges,
validates, and packages the final competition submission.
"""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
SCR = ROOT / "scratch" / "v4"
MODELS = ROOT / "models"
V4 = ROOT / "code" / "business_entity_resolution" / "v4"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def wait_for_model():
    cfg_path = MODELS / "v5_config.json"
    model_path = MODELS / "lgbm_model_v5.pkl"
    log("Waiting for v5 training and calibration to complete...")
    t0 = time.time()
    while True:
        if cfg_path.exists() and model_path.exists():
            log("Found models/v5_config.json and models/lgbm_model_v5.pkl!")
            break
        if time.time() - t0 > 4 * 3600:
            log("TIMEOUT waiting for v5 model artifacts.")
            sys.exit(1)
        time.sleep(15)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    log(f"=== V5 CALIBRATED CONFIG ===")
    for c, info in cfg.get("countries", {}).items():
        log(f"  [{c.upper()}] Expected F0.5: {info.get('f05', 0):.4f} | "
            f"Cap: {info.get('cap')} | Threshold: {info.get('threshold')} | "
            f"MinTop: {info.get('min_top')} | Rescue: {info.get('rescue')}")
    log(f"  Combined Expected F0.5: {cfg.get('combined_expected_f05', 0):.4f}")
    return cfg

def run_parallel_inference():
    log("=== LAUNCHING PARALLEL TEST INFERENCE ===")
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Worker 1: France + US
    log_fw = open(SCR / "test_fw_v5.log", "w", encoding="utf-8")
    cmd_fw = [sys.executable, str(V4 / "run_test.py"), "france,us", "v5_config.json"]
    p_fw = subprocess.Popen(cmd_fw, stdout=log_fw, stderr=subprocess.STDOUT, env=env)
    log(f"Worker 1 (France + US) started [PID {p_fw.pid}] -> scratch/v4/test_fw_v5.log")

    # Worker 2: India
    log_in = open(SCR / "test_in_v5.log", "w", encoding="utf-8")
    cmd_in = [sys.executable, str(V4 / "run_test.py"), "india", "v5_config.json"]
    p_in = subprocess.Popen(cmd_in, stdout=log_in, stderr=subprocess.STDOUT, env=env)
    log(f"Worker 2 (India) started [PID {p_in.pid}] -> scratch/v4/test_in_v5.log")

    # Monitor both
    t0 = time.time()
    while True:
        r_fw = p_fw.poll()
        r_in = p_in.poll()
        if r_fw is not None and r_in is not None:
            break
        elapsed_m = (time.time() - t0) / 60
        log(f"Test inference running... elapsed {elapsed_m:.1f}m (Worker 1: {'DONE' if r_fw is not None else 'RUNNING'}, Worker 2: {'DONE' if r_in is not None else 'RUNNING'})")
        time.sleep(60)

    log_fw.close()
    log_in.close()

    if r_fw != 0:
        log(f"ERROR: Worker 1 exited with code {r_fw}!")
        sys.exit(r_fw)
    if r_in != 0:
        log(f"ERROR: Worker 2 exited with code {r_in}!")
        sys.exit(r_in)
    log("Both inference workers completed successfully!")

def merge_and_package():
    log("=== RUNNING MERGE, VALIDATION & PACKAGING ===")
    
    # 1. Merge & Validate
    cmd_merge = [sys.executable, str(V4 / "merge_validate.py")]
    r = subprocess.run(cmd_merge, capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-1000:], flush=True)
        log("ERROR: merge_validate.py failed!")
        sys.exit(r.returncode)

    # 2. Package Gemz_submission.zip
    build_script = ROOT / "code" / "business_entity_resolution" / "build_submission.py"
    r_pkg = subprocess.run([sys.executable, str(build_script), "--team-name", "Gemz"],
                           capture_output=True, text=True)
    print(r_pkg.stdout[-1000:], flush=True)
    if r_pkg.returncode != 0:
        log("ERROR: build_submission.py failed!")
        sys.exit(r_pkg.returncode)

    # 3. Create convenient matching_results.zip as well
    m_tsv = OUTD / "matching_results.tsv"
    m_zip = OUTD / "matching_results.zip"
    with zipfile.ZipFile(m_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(m_tsv, arcname="matching_results.tsv")
    log(f"Created {m_zip.name} ({m_zip.stat().st_size / (1024*1024):.1f} MB)")
    log("============================================================")
    log("V5 SUBMISSION FULLY PACKAGED & VALIDATED!")
    log(f"Main Zip: Gemz_submission.zip")
    log(f"Matching Results Zip: output/matching_results.zip")
    log("READY TO UPLOAD AS SUBMISSION 4!")
    log("============================================================")

def main():
    wait_for_model()
    run_parallel_inference()
    merge_and_package()

if __name__ == "__main__":
    main()
