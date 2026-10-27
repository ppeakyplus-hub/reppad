"""
Does RepPad help short sequences more than long ones?  Splits users into groups by the length of
their history and compares baseline vs RepPad per group, using the per-user files that
run_seeds.py --save_per_user writes.

    python3 analyze_by_length.py --data ../data/GitHub.txt --name GitHub --seeds 0 1 2 3 4
Output: mean HR@10 / NDCG@10 per length group (mean over seeds), relative gain, number of seeds
where RepPad is higher, and results/by_length_<name>.csv.
"""
import argparse, csv, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--dir", default="results/per_user")
ap.add_argument("--groups", type=int, default=3, help="number of equal-size length groups")
args = ap.parse_args()

seqs = []
for line in open(args.data):
    parts = line.split()
    if parts:
        seqs.append([int(x) for x in parts[1:]])
lengths = np.array([len(x) for x in seqs])
last = np.array([x[-1] for x in seqs])


def per_user(path):
    d = np.load(path)
    users, preds, ans = d["users"], d["preds"], d["answers"].reshape(len(d["users"]), -1)[:, 0]
    best = None
    for off in (0, 1, -1):          # find how user ids index the data file
        idx = users + off
        ok = (idx >= 0) & (idx < len(seqs))
        match = (last[idx[ok]] == ans[ok]).mean() if ok.any() else 0
        if best is None or match > best[1]:
            best = (off, match)
    off, match = best
    if match < 0.99:
        raise SystemExit(f"{path}: true item equals the user's last item for only {match:.0%} of users - "
                         "this file is probably not the test evaluation, or ids differ; stop and check.")
    idx = users + off
    rank = np.array([list(p).index(a) if a in set(p[:20]) else 99 for p, a in zip(preds, ans)])
    hit = (rank < 10).astype(float)
    ndcg = np.where(rank < 10, 1.0 / np.log2(rank + 2), 0.0)
    return lengths[idx], hit, ndcg

res = {0: [], 1: []}
for seed in args.seeds:
    for aug in (0, 1):
        res[aug].append(per_user(f"{args.dir}/{args.name}_aug{aug}_seed{seed}.npz"))

L = res[0][0][0]
cuts = np.quantile(L, np.linspace(0, 1, args.groups + 1))
group = np.clip(np.searchsorted(cuts[1:-1], L, side="right"), 0, args.groups - 1)

rows = []
print(f"{args.name}: {len(L)} users, history length group cuts {np.round(cuts, 1)}")
print(f"{'group':14s}{'users':>6s}  {'HR@10 base':>10s} {'RepPad':>8s} {'gain':>8s}   {'NDCG@10 base':>12s} {'RepPad':>8s} {'gain':>8s}  seeds RepPad>base (HR@10)")
for g in range(args.groups):
    m = group == g
    lo, hi = L[m].min(), L[m].max()
    hb = np.array([r[1][m].mean() for r in res[0]]); hr = np.array([r[1][m].mean() for r in res[1]])
    nb = np.array([r[2][m].mean() for r in res[0]]); nr = np.array([r[2][m].mean() for r in res[1]])
    gh, gn = 100 * (hr.mean() / hb.mean() - 1), 100 * (nr.mean() / nb.mean() - 1)
    wins = int((hr > hb).sum())
    label = f"len {lo}-{hi}"
    print(f"{label:14s}{m.sum():6d}  {hb.mean():10.4f} {hr.mean():8.4f} {gh:+7.1f}%   {nb.mean():12.4f} {nr.mean():8.4f} {gn:+7.1f}%  {wins}/{len(hb)}")
    rows.append([label, m.sum(), hb.mean(), hr.mean(), gh, nb.mean(), nr.mean(), gn, wins, len(hb)])
os.makedirs("results", exist_ok=True)
with open(f"results/by_length_{args.name}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "users", "HR10_base", "HR10_reppad", "HR10_gain_pct", "NDCG10_base", "NDCG10_reppad", "NDCG10_gain_pct", "seeds_reppad_higher", "n_seeds"])
    w.writerows(rows)
print("saved", f"results/by_length_{args.name}.csv")
