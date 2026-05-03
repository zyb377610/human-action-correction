"""快速校准测试"""
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

print("加载视频...")
with FileSource(TEMPLATE) as s: tpl = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST1) as s: t1 = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST2) as s: t2 = estimator.estimate_video(s, show_progress=False)

# 只找1个不同动作
other_seq = None
other_name = ""
for d in glob.glob(r"E:\毕业设计\test-resource\hmdb51_org\*"):
    if "throw" not in d:
        vids = glob.glob(os.path.join(d, "*.avi"))
        if vids:
            try:
                with FileSource(vids[0]) as s:
                    seq = estimator.estimate_video(s, show_progress=False)
                if seq.num_frames > 10:
                    other_seq = seq
                    other_name = os.path.basename(d)
                    break
            except: pass

print(f"\n异类动作: {other_name}")
print(f"\n{'sigma':<8} {'Self':>8} {'T1(同)':>8} {'T2(同)':>8} {'异类':>8} | 区分?")
print("-" * 60)

for sigma in [0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]:
    comp = ActionComparator(algorithm="dtw", metric="euclidean", similarity_sigma=sigma)
    rs = comp.compare(tpl, tpl, "self")
    r1 = comp.compare(t1, tpl, "t1")
    r2 = comp.compare(t2, tpl, "t2")
    rd = comp.compare(other_seq, tpl, "diff") if other_seq else type('', (), {'similarity': 0})()

    min_same = min(r1.similarity, r2.similarity)
    ok = "YES" if rd.similarity < min_same * 0.8 else "NO"
    print(f"{sigma:<8} {rs.similarity:>7.2%} {r1.similarity:>7.2%} {r2.similarity:>7.2%} {rd.similarity:>7.2%} | {ok}")

estimator.close()
print("\n完成!")
