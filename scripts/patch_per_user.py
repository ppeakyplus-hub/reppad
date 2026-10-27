"""
One-off patch for RepPad/SASRec/src/trainers.py: when the environment variable PER_USER_OUT is
set, the full-sort evaluation also saves, for every evaluated user, the user id, the top-20
predicted items and the true item to that .npz file. Training and the reported metrics are
not changed. The test evaluation is the last call of a run, so the file ends up holding the
test predictions (analyze_by_length.py checks this).

    cd RepPad/SASRec/src && python3 patch_per_user.py     # makes trainers.py.bak first
"""
import shutil

P = "trainers.py"
s = open(P).read()
if "PER_USER_OUT" in s:
    raise SystemExit("already patched")

a1 = "                    batch_user_index = user_ids.cpu().numpy()\n"
b1 = a1 + ("                    self._uids = batch_user_index if i == 0 "
           "else np.append(self._uids, batch_user_index)\n")
a2 = "                return self.get_full_sort_score(epoch, answer_list, pred_list)\n"
b2 = ("                _out = __import__('os').environ.get('PER_USER_OUT')\n"
      "                if _out:\n"
      "                    np.savez(_out, users=self._uids, preds=pred_list, answers=answer_list)\n"
      + a2)
assert s.count(a1) == 1 and s.count(a2) == 1, "trainers.py differs from what the patch expects"
shutil.copy(P, P + ".bak")
open(P, "w").write(s.replace(a1, b1).replace(a2, b2))
print("patched trainers.py (backup: trainers.py.bak)")
