"""
Collect GitHub repository-starring sequences, starting from popular repositories.

Why this design: a sample of users chosen by follower count shares almost no repositories
(only 7 of 3,834 repos were starred by 5+ users), so 5-core filtering removes everything.
Here the item vocabulary is fixed first (the most-starred repositories), users are then
sampled from the stargazers of those repositories, and each user's star history is kept
only for repositories in the vocabulary. Items are therefore shared across users.

Steps
  1. vocabulary : top --n_repos repositories by stars          (Search API)
  2. users      : random stargazers of random vocabulary repos (Stargazers API)
  3. sequences  : each user's starred repos with timestamps, restricted to the vocabulary
                  (Starred API, media type star+json gives `starred_at`)

Usage (set GITHUB_TOKEN first, no scopes needed):
    export GITHUB_TOKEN=ghp_xxx
    python3 collect_github_popular.py --n_users 100          # quick test (a few minutes)
    python3 collect_github_popular.py --n_users 3000         # full run (about 1 hour)
The script can be stopped (Ctrl+C) and started again with the same command: it resumes.

Output (folder --workdir, default github_data/):
    stars.csv          user,item,timestamp  -- users are pseudonyms (salted hash), not handles
    vocab.txt          the repositories used as items
    users_queue.txt    GitHub handles still to be processed. LOCAL ONLY, never commit or
                       publish it; the script deletes it when everything is processed.
    .salt              random salt for the pseudonyms. LOCAL ONLY (add to .gitignore).
Then convert with preprocess_github.py.

Rate limits: the script reads the X-RateLimit headers and sleeps until the reset time
when the quota is nearly used up, and backs off on secondary rate limits.
"""

import argparse
import csv
import hashlib
import os
import random
import sys
import time
from urllib.parse import parse_qs, urlparse

import requests

API = "https://api.github.com"


class Client:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "X-GitHub-Api-Version": "2022-11-28"})
        self.n_requests = 0
        self.remaining = None

    def get(self, path, params=None, star=False, retries=6):
        url = path if path.startswith("http") else API + path
        accept = "application/vnd.github.star+json" if star else "application/vnd.github+json"
        r = None
        for attempt in range(retries):
            try:
                r = self.s.get(url, params=params, headers={"Accept": accept}, timeout=30)
            except requests.RequestException as e:
                print(f"  network error ({e}); retrying", flush=True)
                time.sleep(min(2 ** attempt, 60))
                continue
            self.n_requests += 1
            rem = r.headers.get("X-RateLimit-Remaining")
            self.remaining = int(rem) if rem is not None else self.remaining
            if r.status_code in (403, 429):
                text = (r.text or "").lower()
                if r.headers.get("Retry-After"):
                    wait = int(r.headers["Retry-After"]) + 1
                elif rem == "0":
                    wait = max(int(r.headers.get("X-RateLimit-Reset", "0")) - int(time.time()), 0) + 2
                elif "rate limit" in text or "abuse" in text:
                    wait = 60 * (attempt + 1)
                else:
                    return r  # a real 403 (e.g. blocked repository): caller skips it
                print(f"  rate limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            # be nice: if the quota is nearly gone, wait for the reset before the next call
            if self.remaining is not None and self.remaining < 3:
                wait = max(int(r.headers.get("X-RateLimit-Reset", "0")) - int(time.time()), 0) + 2
                print(f"  quota almost used, sleeping {wait}s until reset", flush=True)
                time.sleep(wait)
            return r
        return r


def top_repos(client, n):
    repos, page = [], 1
    while len(repos) < n and page <= 10:  # Search API returns at most 1,000 results
        r = client.get("/search/repositories",
                       {"q": "stars:>1000", "sort": "stars", "order": "desc",
                        "per_page": 100, "page": page})
        if r is None or r.status_code != 200:
            print("search failed:", None if r is None else r.status_code)
            break
        items = r.json().get("items", [])
        if not items:
            break
        repos += [it["full_name"] for it in items]
        page += 1
        time.sleep(2)  # the search endpoint allows about 30 requests per minute
    return repos[:n]


def last_page(resp):
    link = resp.links.get("last")
    if not link:
        return 1
    q = parse_qs(urlparse(link["url"]).query)
    return int(q.get("page", ["1"])[0])


def sample_users(client, repos, n_users, per_repo, rng):
    """Sample ordinary users with the Search Users API, one random day of account creation at a
    time (the stargazers endpoint returned 404 on the author's setup). `repos` is unused."""
    import datetime as dt
    users, seen = [], set()
    start, end = dt.date(2012, 1, 1), dt.date(2024, 12, 31)
    span = (end - start).days
    attempts = 0
    while len(users) < n_users and attempts < 600:
        attempts += 1
        day = start + dt.timedelta(days=rng.randint(0, span))
        q = f"type:user repos:>=10 followers:>=1 created:{day.isoformat()}"
        r = client.get("/search/users", {"q": q, "per_page": 100, "page": 1})
        time.sleep(2.2)  # search endpoints allow about 30 requests per minute
        if r is None or r.status_code != 200:
            print(f"  search failed: {None if r is None else (r.status_code, r.text[:150])}", flush=True)
            continue
        logins = [it["login"] for it in r.json().get("items", []) if it.get("type") == "User"]
        rng.shuffle(logins)
        for login in logins[:per_repo]:
            if login not in seen:
                seen.add(login)
                users.append(login)
        print(f"  sampled users: {len(users)} (accounts created {day})", flush=True)
    return users[:n_users]


def fetch_user_stars(client, login, vocab, max_pages):
    rows = []
    for page in range(1, max_pages + 1):
        r = client.get(f"/users/{login}/starred", {"per_page": 100, "page": page}, star=True)
        if r is None or r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        for e in data:
            name = e["repo"]["full_name"]
            if name in vocab:
                rows.append((name, e["starred_at"]))
        if len(data) < 100:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="github_data")
    ap.add_argument("--n_repos", type=int, default=1000, help="size of the item vocabulary (max 1000)")
    ap.add_argument("--n_users", type=int, default=3000)
    ap.add_argument("--users_per_repo", type=int, default=20, help="users kept per random creation day")
    ap.add_argument("--max_pages", type=int, default=3, help="pages of 100 stars per user (most recent first)")
    ap.add_argument("--min_len", type=int, default=5, help="keep users with at least this many vocabulary stars")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN first:  export GITHUB_TOKEN=ghp_xxx")

    os.makedirs(args.workdir, exist_ok=True)
    p = lambda name: os.path.join(args.workdir, name)
    rng = random.Random(args.seed)
    client = Client(token)

    # salt for pseudonyms (local only)
    if not os.path.exists(p(".salt")):
        with open(p(".salt"), "w") as f:
            f.write(os.urandom(16).hex())
    salt = open(p(".salt")).read().strip()
    pseudo = lambda login: hashlib.sha256((salt + login).encode()).hexdigest()[:12]

    # 1. vocabulary
    if os.path.exists(p("vocab.txt")):
        repos = [l.strip() for l in open(p("vocab.txt")) if l.strip()]
    else:
        print("Step 1: top repositories by stars ...", flush=True)
        repos = top_repos(client, args.n_repos)
        with open(p("vocab.txt"), "w") as f:
            f.write("\n".join(repos) + "\n")
    vocab = set(repos)
    print(f"vocabulary: {len(vocab)} repositories")

    # 2. users
    done_path = p("done_ids.txt")
    done = set(l.strip() for l in open(done_path)) if os.path.exists(done_path) else set()
    if os.path.exists(p("users_queue.txt")):
        queue = [l.strip() for l in open(p("users_queue.txt")) if l.strip()]
    elif done:
        queue = []
    else:
        print("Step 2: sampling users from stargazers ...", flush=True)
        queue = sample_users(client, repos, args.n_users, args.users_per_repo, rng)
        if not queue:
            sys.exit("No users sampled - see the error lines above (token or network problem?).")
        with open(p("users_queue.txt"), "w") as f:
            f.write("\n".join(queue) + "\n")
    todo = [u for u in queue if pseudo(u) not in done]
    print(f"users: {len(queue)} sampled, {len(done)} already processed, {len(todo)} to do")

    # 3. sequences
    new_file = not os.path.exists(p("stars.csv"))
    kept = 0
    with open(p("stars.csv"), "a", newline="") as fcsv, open(done_path, "a") as fdone:
        w = csv.writer(fcsv)
        if new_file:
            w.writerow(["user", "item", "timestamp"])
        for i, login in enumerate(todo, 1):
            rows = fetch_user_stars(client, login, vocab, args.max_pages)
            uid = pseudo(login)
            if len(rows) >= args.min_len:
                for item, ts in rows:
                    w.writerow([uid, item, ts])
                kept += 1
            fdone.write(uid + "\n")
            fcsv.flush()
            fdone.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} users processed, {kept} kept, "
                      f"requests {client.n_requests}, quota left {client.remaining}", flush=True)

    if os.path.exists(p("users_queue.txt")):
        os.remove(p("users_queue.txt"))
        print("All users processed; deleted users_queue.txt (it contained GitHub handles).")
    print(f"Done. Next: python3 preprocess_github.py --input {p('stars.csv')} --output GitHub.txt --k 5")


if __name__ == "__main__":
    main()
