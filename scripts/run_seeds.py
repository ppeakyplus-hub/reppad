"""
Run RepPad (SASRec) baseline vs RepPad over several seeds and save one CSV row per run.

Put this file in RepPad/SASRec/src (next to main.py) and run from that folder:

    python3 run_seeds.py --data_name LastFM --seeds 0 1 2
    python3 run_seeds.py --data_name Beauty --seeds 0 1 2 --extra "--epochs 50"

For each seed it runs two settings:
    aug_type=0  baseline (zero padding)
    aug_type=1  RepPad
and records the test metrics and the wall-clock time of the run.

Output:
    results/runs.csv        one row per run (appended; finished runs are skipped on re-run)
    results/logs/*.log      full console output of every run (check parsing against these)
Then prints mean +- std per (dataset, setting) and the relative gain of RepPad.

Note: metrics are read from the LAST occurrence of HIT@5/NDCG@5/HIT@10/NDCG@10 in the
console output, which should be the final test evaluation. Check one log file by eye
the first time to make sure the numbers match what the program prints as "test".
"""

import argparse
import csv
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

METRICS = ["HIT@5", "NDCG@5", "HIT@10", "NDCG@10"]
FIELDS = ["data_name", "aug_type", "setting", "seed", "model_idx", *METRICS, "seconds", "returncode"]
SETTING = {0: "baseline", 1: "reppad"}


def parse_metrics(text):
    """Take the last value printed for each metric (formats like HIT@5=0.04 or 'HIT@5': '0.04')."""
    found = {}
    for name in METRICS:
        pattern = re.escape(name) + r"""['"]?\s*[:=]\s*['"]?([0-9]*\.?[0-9]+)"""
        values = re.findall(pattern, text)
        found[name] = float(values[-1]) if values else None
    return found


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["returncode"] == "0" and row[METRICS[0]] != "":
                    done.add((row["data_name"], int(row["aug_type"]), int(row["seed"])))
    return done


def summarise(path, data_name):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["data_name"] == data_name and row["returncode"] == "0" and row[METRICS[0]] != "":
                rows.append(row)
    print(f"\n=== {data_name}: mean +- std over seeds ===")
    means = {}
    for aug in (0, 1):
        sel = [r for r in rows if int(r["aug_type"]) == aug]
        if not sel:
            continue
        line = f"{SETTING[aug]:9s} (n={len(sel)}) "
        means[aug] = {}
        for m in METRICS:
            vals = [float(r[m]) for r in sel]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            means[aug][m] = mean
            line += f"{m}={mean:.4f}+-{sd:.4f}  "
        secs = [float(r["seconds"]) for r in sel]
        line += f"time/run={statistics.mean(secs):.0f}s"
        print(line)
    if 0 in means and 1 in means:
        gains = "  ".join(f"{m}: {100 * (means[1][m] / means[0][m] - 1):+.1f}%" for m in METRICS)
        print(f"RepPad gain over baseline: {gains}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--extra", default="", help='extra args for main.py, e.g. "--epochs 50"')
    ap.add_argument("--out", default="results/runs.csv")
    ap.add_argument("--save_per_user", action="store_true",
                    help="also save per-user predictions to results/per_user/ (needs patch_per_user.py)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    done = load_done(args.out)
    new_file = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for seed in args.seeds:
            for aug in (0, 1):
                if (args.data_name, aug, seed) in done:
                    print(f"skip {args.data_name} aug={aug} seed={seed} (already done)")
                    continue
                # model_idx names the checkpoint file, so give every run its own to avoid overwriting
                model_idx = 1000 + seed * 10 + aug
                cmd = [sys.executable, "main.py", f"--data_name={args.data_name}",
                       f"--aug_type={aug}", f"--model_idx={model_idx}", f"--seed={seed}",
                       *shlex.split(args.extra)]
                print("running:", " ".join(cmd), flush=True)
                t0 = time.time()
                env = dict(os.environ)
                if args.save_per_user:
                    os.makedirs("results/per_user", exist_ok=True)
                    env["PER_USER_OUT"] = f"results/per_user/{args.data_name}_aug{aug}_seed{seed}.npz"
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                secs = time.time() - t0
                output = proc.stdout + "\n" + proc.stderr
                with open(f"results/logs/{args.data_name}_aug{aug}_seed{seed}.log", "w") as lf:
                    lf.write(output)
                m = parse_metrics(output)
                row = {"data_name": args.data_name, "aug_type": aug, "setting": SETTING[aug],
                       "seed": seed, "model_idx": model_idx, "seconds": f"{secs:.1f}",
                       "returncode": proc.returncode}
                row.update({k: ("" if v is None else v) for k, v in m.items()})
                writer.writerow(row)
                f.flush()
                print(f"  done in {secs:.0f}s rc={proc.returncode} " +
                      " ".join(f"{k}={m[k]}" for k in METRICS), flush=True)
                if proc.returncode != 0 or m[METRICS[0]] is None:
                    print("  WARNING: run failed or metrics not found - open the log in results/logs/")

    summarise(args.out, args.data_name)


if __name__ == "__main__":
    main()
