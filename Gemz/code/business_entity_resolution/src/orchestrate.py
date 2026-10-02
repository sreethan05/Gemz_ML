"""v8 orchestrator v3 (serial, memory-safe): wait for US artifacts -> India ->
france config copies (france_uses=us per evidence) -> Phase C -> best rules."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "dataset" / "test").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
while not (ROOT / 'dataset' / 'test').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
CONFIGS = ROOT / "scratch" / "v8"
if not CONFIGS.exists():
    CONFIGS = Path(__file__).resolve().parent / "configs"
while not (ROOT / 'dataset' / 'test').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

SCR8 = CONFIGS


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def run_logged(cmd, logname):
    with open(SCR8 / logname, "w", encoding="utf-8") as lf:
        r = subprocess.run([sys.executable, *cmd], stdout=lf, stderr=subprocess.STDOUT)
    tail = (SCR8 / logname).read_text(encoding="utf-8").strip().splitlines()[-6:]
    print("\n".join(tail), flush=True)
    return r.returncode


def main():
    log("waiting for US artifacts (us_train_X.npy) ...")
    while not (SCR8 / "us_train_X.npy").exists():
        time.sleep(60)
    time.sleep(30)  # let US finish writing/saving
    log("US done -> starting India (solo)")
    rc = run_logged(["scratch/v8/retrieval_v8.py", "india"], "run_india.log")
    if rc != 0:
        log(f"INDIA FAILED rc={rc}")
        sys.exit(2)

    shutil.copy(SCR8 / "us_retrieval_ab.json", SCR8 / "france_retrieval_ab.json")
    log("france config copy (france_uses=us)")

    rc = run_logged(["scratch/v8/phase_c.py"], "phase_c.log")
    if rc != 0:
        log(f"PHASE C FAILED rc={rc}")
        sys.exit(4)

    with open(SCR8 / "calibration_report.json") as f:
        rep = json.load(f)
    for c in ("us", "india"):
        best = None
        for tag in ("n3", "n2", "n1", "v4base"):
            for m in ("ens", "m1", "m2"):
                k = f"{c}|{tag}|{m}"
                if k in rep and (best is None or rep[k]["f05"] > best[0]):
                    best = (rep[k]["f05"], tag, m, rep[k]["th"], rep[k]["mt"])
        if best:
            with open(SCR8 / f"{c}_best_rule.json", "w") as f:
                json.dump({"f05": best[0], "tag": best[1], "model": best[2],
                           "th": best[3], "mt": best[4]}, f, indent=1)
            log(f"{c} BEST: {best}")
    shutil.copy(SCR8 / "us_best_rule.json", SCR8 / "france_best_rule.json")
    log("ORCHESTRATION COMPLETE")


if __name__ == "__main__":
    main()
