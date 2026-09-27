"""Endgame chain: wait for orchestrator (india+phase_c+rules) -> pass-2 for
winners outside the saved feats set -> addon calibration -> france=us rule ->
staggered parallel Phase D -> merge + official validate."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCR8 = ROOT / "scratch" / "v8"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def run_logged(cmd, logname):
    with open(SCR8 / logname, "w", encoding="utf-8") as lf:
        r = subprocess.run([sys.executable, *cmd], stdout=lf, stderr=subprocess.STDOUT)
    tail = (SCR8 / logname).read_text(encoding="utf-8").strip().splitlines()[-8:]
    print("\n".join(tail), flush=True)
    return r.returncode


def saved_tags(c):
    return [t for t in ("n1", "n2", "n3", "n4", "n5")
            if (SCR8 / f"{c}_feats_{t}.npy").exists()]


def main():
    log("waiting for orchestrator to complete (india + phase_c + rules) ...")
    while True:
        try:
            txt = (SCR8 / "orchestrate.log").read_text(encoding="utf-8")
        except OSError:
            txt = ""
        if "ORCHESTRATION COMPLETE" in txt:
            break
        if "FAILED" in txt:
            log("orchestrator failed — abort")
            sys.exit(2)
        time.sleep(30)
    log("orchestrator done")

    # pass-2 for winners whose feats were not saved
    for c in ("us", "india"):
        with open(SCR8 / f"{c}_retrieval_ab.json") as f:
            best = json.load(f)["best"]
        if best not in saved_tags(c):
            log(f"{c}: winner {best} lacks feats -> pass2")
            rc = run_logged(["scratch/v8/pass2_gen.py", c, best], f"pass2_{c}.log")
            if rc != 0:
                log(f"pass2 {c} FAILED rc={rc}")
                sys.exit(3)

    # addon calibration for winner tags missing from the report
    with open(SCR8 / "calibration_report.json") as f:
        rep = json.load(f)
    for c in ("us", "india"):
        with open(SCR8 / f"{c}_retrieval_ab.json") as f:
            best = json.load(f)["best"]
        for m in ("m1", "m2", "ens"):
            if f"{c}|{best}|{m}" not in rep:
                log(f"addon calibration {c}|{best}")
                rc = run_logged(["scratch/v8/phase_c_addon.py", c, best],
                                f"addon_{c}.log")
                break

    # france = us rule
    shutil.copy(SCR8 / "us_best_rule.json", SCR8 / "france_best_rule.json")
    log("france rule = us rule")

    # parallel phase D
    rc = run_logged(["scratch/v8/phase_d_parallel.py"], "phase_d.log")
    if rc != 0:
        log(f"PHASE D FAILED rc={rc}")
        sys.exit(4)

    # merge + official validation
    rc = run_logged(["scratch/v8/finish_v8.py"], "finish.log")
    log(f"finish rc={rc}")
    log("CHAIN COMPLETE" if rc == 0 else "CHAIN DONE WITH VALIDATION ISSUES")


if __name__ == "__main__":
    main()
