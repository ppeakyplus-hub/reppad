# reppad
## Notes on the result files
- `results/runs.csv`: first set of runs (all datasets). `results/runs_peruser.csv`: later re-runs of GitHub and LastFM seeds 0-4 made to save per-user predictions.
- Because of these re-runs, the logs in `results/logs/` for GitHub and LastFM seeds 0-4 are from the re-runs, not from the first runs. Training was not fully deterministic, so numbers for the same seed differ between the two sets (see the report).
- `data/GitHub_sample_100users.txt` holds the first 100 users of the final dataset (indexed, no usernames); the full user-level file is not published.
