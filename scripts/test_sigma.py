"""校准 sigma 参数"""
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
TEST2 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_2.avi"

estimator = PoseEstimator()
with FileSource(TEMPLATE) as s: tpl = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST1) as s: t1 = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST2) as s: t2 = estimator.estimate_video(s, show_progress=False)

# 加载不同动作
other_dirs = glob.glob(r"E:\毕业设计\test-resource\hmdb51_org\*")
others = []
for d in other_dirs:
    if "throw" not in d:
        vids = glob.glob(os.path.join(d, "*.avi"))
        if vids:
            try:
                with FileSource(vids[0]) as s:
                    seq = estimator.estimate_video(s, show_progress=False)
                if seq.num_frames > 10:
                    others.append((os.path.basename(d), seq))
            except: pass
    if len(others) >= 3:
        break

def score_map(sim):
    if sim >= 0.999: return 100.0
    elif sim >= 0.85: return 90.0 + (sim - 0.85) / 0.15 * 10.0
    elif sim >= 0.65: return 65.0 + (sim - 0.65) / 0.2 * 25.0
    elif sim >= 0.4: return 35.0 + (sim - 0.4) / 0.25 * 30.0
    elif sim >= 0.15: return 10.0 + (sim - 0.15) / 0.25 * 25.0
    else: return sim / 0.15 * 10.0

print(f"{'sigma':<8} {'Self':>8} {'T1(同)':>8} {'T2(同)':>8}", end="")
for name, _ in others:
    print(f" {name[:8]+'(异)':>12}", end="")
print(f"  | {'T1分':>6} {'T2分':>6} {'异最高分':>8} {'能区分':>6}")
print("-" * 120)

for sigma in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]:
    comp = ActionComparator(algorithm="dtw", metric="euclidean", similarity_sigma=sigma)
    rs = comp.compare(tpl, tpl, "self")
    r1 = comp.compare(t1, tpl, "t1")
    r2 = comp.compare(t2, tpl, "t2")

    diff_sims = []
    diff_strs = []
    for name, seq in others:
        r = comp.compare(seq, tpl, name)
        diff_sims.append(r.similarity)
        diff_strs.append(f"{r.similarity:>11.2%}")

    max_diff = max(diff_sims) if diff_sims else 0
    min_same = min(r1.similarity, r2.similarity)
    ok = "YES" if max_diff < min_same * 0.8 else "NO"

    print(f"{sigma:<8} {rs.similarity:>7.2%} {r1.similarity:>7.2%} {r2.similarity:>7.2%}", end="")
    for ds in diff_strs:
        print(f" {ds}", end="")
    print(f"  | {score_map(r1.similarity):>5.1f} {score_map(r2.similarity):>5.1f} {score_map(max_diff):>7.1f} {ok:>6}")

estimator.close()
