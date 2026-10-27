"""
Convert the GitHub star CSV (user,item,timestamp) into the text format RepPad reads
(the same as Beauty.txt, LastFM.txt, ...): one line per user,

    <user_id> <item_id> <item_id> ...

with user ids 1..U, item ids 1..I (0 is reserved for padding) and each user's items
in chronological order.  The paper's datasets use 5-core filtering, so users and items
with fewer than k interactions are removed repeatedly until nothing changes.

Usage:
    python3 preprocess_github.py --input github_star_sequences_scaled.csv \
        --output GitHub.txt --k 5

Outputs:
    GitHub.txt            data file in RepPad format (copy it into RepPad/SASRec/data/)
    GitHub_item_map.csv   item_id,repository  (repositories are public names)
Usernames are NOT written anywhere: users are only numbered, so the released
file does not contain GitHub handles.
"""

import argparse
import csv
from collections import defaultdict


def k_core(seqs, k):
    """seqs: {user: [(timestamp, item), ...]}. Remove users/items with < k interactions until stable."""
    while True:
        item_count = defaultdict(int)
        for events in seqs.values():
            for _, item in events:
                item_count[item] += 1
        new = {}
        for user, events in seqs.items():
            kept = [(t, i) for t, i in events if item_count[i] >= k]
            if len(kept) >= k:
                new[user] = kept
        if sum(len(v) for v in new.values()) == sum(len(v) for v in seqs.values()):
            return new
        seqs = new


def stats(seqs):
    users = len(seqs)
    items = len({i for ev in seqs.values() for _, i in ev})
    inter = sum(len(ev) for ev in seqs.values())
    avg = inter / users if users else 0
    sparsity = 1 - inter / (users * items) if users and items else 1
    return users, items, inter, avg, sparsity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="GitHub.txt")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    seqs = defaultdict(list)
    with open(args.input, newline="") as f:
        for row in csv.DictReader(f):
            seqs[row["user"]].append((row["timestamp"], row["item"]))
    # a user can star a repository only once, but drop duplicates defensively
    for u in seqs:
        seen, uniq = set(), []
        for t, i in sorted(seqs[u]):
            if i not in seen:
                seen.add(i)
                uniq.append((t, i))
        seqs[u] = uniq

    u0, i0, n0, a0, s0 = stats(seqs)
    print(f"raw        : {u0} users, {i0} items, {n0} interactions, avg len {a0:.1f}, sparsity {s0:.4f}")

    core = k_core(dict(seqs), args.k)
    u1, i1, n1, a1, s1 = stats(core)
    print(f"{args.k}-core     : {u1} users, {i1} items, {n1} interactions, avg len {a1:.1f}, sparsity {s1:.4f}")
    if not core:
        print("Nothing survives this filter: the sample is too sparse (items are not shared "
              "between enough users). Collect more users or use a smaller k.")
        return

    item_id = {}
    with open(args.output, "w") as out:
        for uid, user in enumerate(sorted(core), start=1):
            ids = []
            for _, item in core[user]:
                if item not in item_id:
                    item_id[item] = len(item_id) + 1
                ids.append(str(item_id[item]))
            out.write(f"{uid} {' '.join(ids)}\n")
    map_path = args.output.rsplit(".", 1)[0] + "_item_map.csv"
    with open(map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "repository"])
        for item, i in item_id.items():
            w.writerow([i, item])
    print(f"wrote {args.output} and {map_path}")


if __name__ == "__main__":
    main()
