"""
LLM-as-judge for the GitHub data: is a recommendation list relevant to the user's starring history?

For a random sample of test users it builds, for the baseline and for RepPad, a prompt with the
user's last N starred repositories and the model's top-10 recommended repositories (names only),
asks a local open-source model (Ollama) to decide, for each recommended repository, whether it is
relevant to the developer's interests (yes/no), and compares the share of relevant items.
Each list is judged separately, with no mention of which model produced it, and the judge runs at
temperature 0 with a fixed seed. The model digest is recorded so the version is fixed.

Setup (once):  install Ollama (https://ollama.com), then   ollama pull qwen2.5:7b
Run (from RepPad/SASRec/src, after run_seeds.py --save_per_user on GitHub):
    python3 judge_github.py --seed 0 --n_users 150 --model qwen2.5:7b
Quick test of the pipeline without a model:  --backend mock
Outputs: results/judge_github_seed<seed>.csv (per user scores) and a summary printed at the end.
"""
import argparse, csv, json, os, random, re, sys, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data/GitHub.txt")
ap.add_argument("--item_map", default="GitHub_item_map.csv")
ap.add_argument("--dir", default="results/per_user")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_users", type=int, default=150)
ap.add_argument("--history", type=int, default=15, help="how many of the latest starred repos to show")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--model", default="qwen2.5:7b")
ap.add_argument("--control", action="store_true",
                help="validity check: can the judge tell 10 repos the user really starred later from 10 random repos?")
ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
ap.add_argument("--host", default="http://localhost:11434")
args = ap.parse_args()

names = {int(r["item_id"]): r["repository"] for r in csv.DictReader(open(args.item_map))}
seqs = [[int(x) for x in l.split()[1:]] for l in open(args.data) if l.strip()]


def load(aug):
    d = np.load(f"{args.dir}/GitHub_aug{aug}_seed{args.seed}.npz")
    return d["users"], d["preds"], d["answers"].reshape(len(d["users"]), -1)[:, 0]

u0, p0, a0 = load(0)
u1, p1, a1 = load(1)
assert (u0 == u1).all(), "different user order in the two runs"
off = 0 if (np.array([seqs[u][-1] for u in u0]) == a0).mean() > 0.99 else 1
assert (np.array([seqs[u + off][-1] for u in u0]) == a0).mean() > 0.99, "cannot align users with the data file"

rng = random.Random(0)
pick = sorted(rng.sample(range(len(u0)), min(args.n_users, len(u0))))

PROMPT = """You are evaluating a recommender system for GitHub repositories.

A developer recently starred these repositories (oldest to newest):
{hist}

The system recommends these {k} repositories:
{recs}

For EACH recommended repository decide whether this developer would plausibly be interested in it:
say "yes" if it is in an area the developer has shown interest in (same language, ecosystem, kind of
tool or topic) or is a natural next thing for such a developer to look at; say "no" if it is in an
unrelated area. Consider each repository separately and give a reason that names it. Judge from the
repository names and your general knowledge of them. Do not say "yes" just because a repository is
popular.
Answer with JSON only, with one entry per recommended repository in the order shown:
{{"judgements": [{{"n": 1, "relevant": "yes" or "no", "why": "<few words>"}}, ...]}}"""


def build(user_row, preds):
    hist = seqs[user_row + off][:-2][-args.history:]  # history before the validation and test items
    h = "\n".join(f"- {names[i]}" for i in hist)
    recs = "\n".join(f"{j + 1}. {names[int(i)]}" for j, i in enumerate(preds[:args.k]))
    return PROMPT.format(hist=h, recs=recs, k=args.k)


def ask(prompt):
    """Return (share of recommended items judged relevant, short text)."""
    if args.backend == "mock":
        r = random.Random(hash(prompt) % 10**6)
        return sum(r.random() < 0.3 for _ in range(args.k)) / args.k, "mock"
    body = json.dumps({"model": args.model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "seed": 0}}).encode()
    req = urllib.request.Request(args.host + "/api/generate", body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]
    try:
        items = json.loads(out)["judgements"]
        flags = [str(x.get("relevant", "")).strip().lower().startswith("y") for x in items]
        if len(flags) != args.k:
            return None, f"expected {args.k} judgements, got {len(flags)}"
        why = "; ".join(f"{int(f)}:{str(x.get('why', ''))[:25]}" for f, x in zip(flags, items))
        return sum(flags) / args.k, why
    except Exception:
        return None, out[:80]


PAIR_PROMPT = """You are evaluating two recommender systems for GitHub repositories.

A developer recently starred these repositories (oldest to newest):
{hist}

List A:
{a}

List B:
{b}

Which list is more relevant to this developer's interests (same languages, ecosystems, kinds of
tool or topics)? Judge from the repository names and your general knowledge of them. Do not prefer
a list just because it is more popular or comes first.
Answer with JSON only: {{"better": "A" or "B" or "tie", "why": "<one short sentence>"}}"""


def ask_pair(prompt):
    if args.backend == "mock":
        return random.Random(hash(prompt) % 10**6).choice(["A", "B", "tie"])
    body = json.dumps({"model": args.model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "seed": 0}}).encode()
    req = urllib.request.Request(args.host + "/api/generate", body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]
    try:
        v = str(json.loads(out)["better"]).strip().upper()
        return v if v in ("A", "B") else "tie"
    except Exception:
        return None


if args.backend == "ollama":
    tags = json.loads(urllib.request.urlopen(args.host + "/api/tags").read())["models"]
    dig = [t["digest"][:12] for t in tags if t["name"].startswith(args.model)]
    print("judge model:", args.model, "digest:", dig[0] if dig else "unknown (is the model pulled?)")


if args.control:
    # positive control, independent of any recommender: users with long histories; history = the part
    # before their last 12 stars; list C = their last 10 real stars; list D = 10 random repositories
    # they never starred. A valid judge should clearly prefer C.
    all_items = sorted(names)
    cands = [u for u in range(len(seqs)) if len(seqs[u]) >= args.history + 12]
    rr = random.Random(1)
    chosen = rr.sample(cands, min(args.n_users, len(cands)))
    prefs = []
    for n, u in enumerate(chosen, 1):
        sq = seqs[u]
        hist = "\n".join(f"- {names[x]}" for x in sq[:-12][-args.history:])
        real = sq[-10:]
        pool = [x for x in all_items if x not in set(sq)]
        rand = rr.sample(pool, 10)
        lc = "\n".join(f"{j + 1}. {names[x]}" for j, x in enumerate(real))
        ld = "\n".join(f"{j + 1}. {names[x]}" for j, x in enumerate(rand))
        v1 = ask_pair(PAIR_PROMPT.format(hist=hist, a=lc, b=ld))   # A = real
        v2 = ask_pair(PAIR_PROMPT.format(hist=hist, a=ld, b=lc))   # A = random
        if v1 is None or v2 is None:
            continue
        pts = {"A": 1.0, "B": 0.0, "tie": 0.5}
        prefs.append((pts[v1] + (1 - pts[v2] if v2 != "tie" else 0.5)) / 2)
        if n % 10 == 0:
            print(f"  control: judged {n}/{len(chosen)} users", flush=True)
    prefs = np.array(prefs)
    print(f"\nVALIDITY CHECK ({len(prefs)} users): judge prefers the REAL later stars over random repositories "
          f"{100 * prefs.mean():.1f}% of the time (50% = judge cannot tell them apart); "
          f"consistent wins {int((prefs == 1).sum())}, consistent losses {int((prefs == 0).sum())}, "
          f"mixed/tie {int(((prefs > 0) & (prefs < 1)).sum())}")
    sys.exit(0)

os.makedirs("results", exist_ok=True)
rows = []
for n, i in enumerate(pick, 1):
    row = {"user_row": int(u0[i])}
    for tag, preds, ans in (("baseline", p0, a0), ("reppad", p1, a1)):
        score, reason = ask(build(int(u0[i]), preds[i]))
        row[f"{tag}_score"] = score
        row[f"{tag}_hit10"] = int(ans[i] in set(int(x) for x in preds[i][:10]))
        row[f"{tag}_reason"] = reason
    # pairwise preference, both orders (A/B swapped) to cancel position bias
    hist = "\n".join(f"- {names[x]}" for x in seqs[int(u0[i]) + off][:-2][-args.history:])
    lb = "\n".join(f"{j + 1}. {names[int(x)]}" for j, x in enumerate(p0[i][:args.k]))
    lr = "\n".join(f"{j + 1}. {names[int(x)]}" for j, x in enumerate(p1[i][:args.k]))
    v1 = ask_pair(PAIR_PROMPT.format(hist=hist, a=lb, b=lr))   # A = baseline, B = RepPad
    v2 = ask_pair(PAIR_PROMPT.format(hist=hist, a=lr, b=lb))   # A = RepPad, B = baseline
    if v1 is None or v2 is None:
        row["reppad_pref"] = None
    else:
        pts = {"B": 1.0, "A": 0.0, "tie": 0.5}
        row["reppad_pref"] = (pts[v1] + (1 - pts[v2] if v2 != "tie" else 0.5)) / 2
    rows.append(row)
    if n % 10 == 0:
        print(f"  judged {n}/{len(pick)} users", flush=True)

out = f"results/judge_github_seed{args.seed}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["baseline_score"] is not None and r["reppad_score"] is not None]
print(f"\nvalid pairs: {len(ok)}/{len(rows)}")
b = np.array([r["baseline_score"] for r in ok], float)
r_ = np.array([r["reppad_score"] for r in ok], float)
print(f"mean share of relevant items  baseline {b.mean():.3f}   RepPad {r_.mean():.3f}   (difference {r_.mean() - b.mean():+.3f})")
print(f"RepPad higher for {int((r_ > b).sum())} users, lower for {int((r_ < b).sum())}, equal for {int((r_ == b).sum())}")
try:
    from scipy import stats
    print("Wilcoxon signed-rank p =", round(stats.wilcoxon(r_, b).pvalue, 4) if (r_ != b).any() else "n/a (no differences)")
    scores = np.concatenate([b, r_]); hits = np.array([r["baseline_hit10"] for r in ok] + [r["reppad_hit10"] for r in ok], float)
    if hits.std() > 0:
        print("agreement with HR@10 (Spearman, share of relevant items vs hit):", round(stats.spearmanr(scores, hits)[0], 3),
              f"[hits: {int(hits.sum())} of {len(hits)} lists]")
except Exception as e:
    print("(scipy not available or test failed:", e, ")")
pr = [r["reppad_pref"] for r in rows if r.get("reppad_pref") is not None]
if pr:
    pr = np.array(pr)
    print(f"\npairwise (both orders): RepPad list preferred by judge {100 * pr.mean():.1f}% of the time "
          f"(50% = no preference); consistent RepPad wins {int((pr == 1).sum())}, consistent baseline wins "
          f"{int((pr == 0).sum())}, mixed/tie {int(((pr > 0) & (pr < 1)).sum())}  (n={len(pr)})")
    try:
        from scipy import stats
        wins, losses = int((pr == 1).sum()), int((pr == 0).sum())
        if wins + losses:
            print("sign test on consistent wins/losses: p =", round(stats.binomtest(wins, wins + losses, 0.5).pvalue, 4))
    except Exception as e:
        print("(sign test failed:", e, ")")
print("saved", out)
