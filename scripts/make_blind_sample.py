"""
Build a BLIND sample for an outside judge (a person or an LLM that must not know which list came from
which system). Two kinds of comparisons are shuffled together:
  model    : baseline top-10 vs RepPad top-10 for the same user
  control  : the user's real last 10 stars vs 10 random repositories (a judge that is valid must pick the real ones)
Writes   results/blind_sample.md   (send THIS file to the judge; it contains no labels)
         results/blind_key.csv     (keep this; do not send it until the judge has answered)

    python3 make_blind_sample.py --n_model 30 --n_control 30
"""
import argparse, csv, random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data/GitHub.txt")
ap.add_argument("--item_map", default="GitHub_item_map.csv")
ap.add_argument("--dir", default="results/per_user")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_model", type=int, default=30)
ap.add_argument("--n_control", type=int, default=30)
ap.add_argument("--history", type=int, default=15)
args = ap.parse_args()

names = {int(r["item_id"]): r["repository"] for r in csv.DictReader(open(args.item_map))}
seqs = [[int(x) for x in l.split()[1:]] for l in open(args.data) if l.strip()]
d0 = np.load(f"{args.dir}/GitHub_aug0_seed{args.seed}.npz")
d1 = np.load(f"{args.dir}/GitHub_aug1_seed{args.seed}.npz")
users, p0, p1 = d0["users"], d0["preds"], d1["preds"]
ans = d0["answers"].reshape(len(users), -1)[:, 0]
off = 0 if (np.array([seqs[u][-1] for u in users]) == ans).mean() > 0.99 else 1
assert (np.array([seqs[u + off][-1] for u in users]) == ans).mean() > 0.99

rng = random.Random(123)
items = []
for i in rng.sample(range(len(users)), args.n_model):
    sq = seqs[int(users[i]) + off]
    items.append(dict(kind="model", hist=sq[:-2][-args.history:],
                      base=[int(x) for x in p0[i][:10]], rep=[int(x) for x in p1[i][:10]]))
cands = [u for u in range(len(seqs)) if len(seqs[u]) >= args.history + 12]
vocab = sorted(names)
for u in rng.sample(cands, args.n_control):
    sq = seqs[u]
    pool = [x for x in vocab if x not in set(sq)]
    items.append(dict(kind="control", hist=sq[:-12][-args.history:], base=sq[-10:], rep=rng.sample(pool, 10)))
rng.shuffle(items)

md = ["# Blind judging sample", "",
      "For each case read the developer's starred repositories (oldest to newest), then decide which list "
      "(A or B) is more relevant to that developer's interests: same languages, ecosystems, kinds of tool or topics. "
      "Judge from repository names and general knowledge. Do not prefer a list for being more popular or for its position. "
      "Answer `A`, `B` or `tie` for every case.", ""]
key = []
for n, it in enumerate(items, 1):
    swap = rng.random() < 0.5            # base/rep (or real/random) goes to A or B at random
    first, second = (it["rep"], it["base"]) if swap else (it["base"], it["rep"])
    a_is = ("rep" if it["kind"] == "model" else "random") if swap else ("base" if it["kind"] == "model" else "real")
    md += [f"## Case {n}", "**Starred:** " + "; ".join(names[x] for x in it["hist"]), "",
           "**List A:** " + "; ".join(names[x] for x in first),
           "**List B:** " + "; ".join(names[x] for x in second), ""]
    key.append([n, it["kind"], a_is])
open("results/blind_sample.md", "w").write("\n".join(md))
with open("results/blind_key.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "kind", "A_is"]); w.writerows(key)
print(f"wrote results/blind_sample.md ({len(items)} cases) and results/blind_key.csv - send only the .md to the judge")
