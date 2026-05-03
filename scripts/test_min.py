"""最小化验证：角度特征的区分度 + sigma校准"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np

from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.video_source import FileSource
from src.action_comparison.comparison import ActionComparator

TEMPLATE = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_0.avi"
TEST1 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_1.avi"

estimator = PoseEstimator()
print("Loading template...", flush=True)
with FileSource(TEMPLATE) as s: tpl = estimator.estimate_video(s, show_progress=False)
print("Loading test1...", flush=True)
with FileSource(TEST1) as s: t1 = estimator.estimate_video(s, show_progress=False)

# 找1个不同动作
print("Loading different action...", flush=True)
diff_seq = None
diff_name = ""
for d in sorted(glob.glob(r"E:\毕业设计\test-resource\hmdb51_org\*")):
    if "throw" not in os.path.basename(d):
        vids = glob.glob(os.path.join(d, "*.avi"))
        if vids:
            try:
                with FileSource(vids[0]) as s:
                    seq = estimator.estimate_video(s, show_progress=False)
                if seq.num_frames > 10:
                    diff_seq = seq
                    diff_name = os.path.basename(d)
                    print(f"  Found: {diff_name} ({seq.num_frames} frames)", flush=True)
                    break
            except:
                pass

print("\nRunning comparisons...", flush=True)
for sigma in [0.5, 0.6, 0.7, 0.8]:
    comp = ActionComparator(algorithm="dtw", metric="euclidean", similarity_sigma=sigma)
    r_self = comp.compare(tpl, tpl, "self")
    r_same = comp.compare(t1, tpl, "same")
    r_diff = comp.compare(diff_seq, tpl, "diff") if diff_seq else None

    diff_str = f"{r_diff.similarity:.2%}" if r_diff else "N/A"
    gap_ok = r_diff and (r_diff.similarity < r_same.similarity * 0.8)
    print(f"  sigma={sigma}: self={r_self.similarity:.2%}, same_action={r_same.similarity:.2%}, diff_action({diff_name})={diff_str}, distinguish={'YES' if gap_ok else 'NO'}", flush=True)

estimator.close()
print("Done!", flush=True)
