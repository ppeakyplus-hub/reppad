"""
Score the blind judging. answers.csv has columns case,choice  (choice = A, B or tie).
    python3 score_blind.py --answers results/blind_answers.csv
Reports (1) validity: how often the judge picked the REAL stars over random repositories (control cases);
(2) baseline vs RepPad preference on the model cases, with a sign test (ties are ignored).
"""
import argparse, csv
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
ap.add_argument("--key", default="results/blind_key.csv")
args = ap.parse_args()
key = {int(r["case"]): r for r in csv.DictReader(open(args.key))}
ctl, mod = [], []
for r in csv.DictReader(open(args.answers)):
    k = key[int(r["case"])]; c = r["choice"].strip().upper()
    pick = "A" if c == "A" else "B" if c == "B" else "tie"
    if k["kind"] == "control":
        ctl.append("tie" if pick == "tie" else ("real" if (pick == "A") == (k["A_is"] == "real") else "random"))
    else:
        mod.append("tie" if pick == "tie" else ("rep" if (pick == "A") == (k["A_is"] == "rep") else "base"))
w, l, t = ctl.count("real"), ctl.count("random"), ctl.count("tie")
print(f"VALIDITY (control, n={len(ctl)}): real stars chosen {w}, random chosen {l}, tie {t}")
if w + l: print("  accuracy excluding ties:", f"{w / (w + l):.1%}", " sign test p =", round(stats.binomtest(w, w + l, 0.5).pvalue, 4))
rw, bw, mt = mod.count("rep"), mod.count("base"), mod.count("tie")
print(f"MODEL cases (n={len(mod)}): RepPad preferred {rw}, baseline preferred {bw}, tie {mt}")
if rw + bw: print("  RepPad share excluding ties:", f"{rw / (rw + bw):.1%}", " sign test p =", round(stats.binomtest(rw, rw + bw, 0.5).pvalue, 4))
