"""快速对比不同算法"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.WARNING)

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

print(f"{'Algorithm':<12} {'Metric':<12} {'Self':>8} {'Test1':>8} {'Test2':>8} {'T1_dist':>10} {'T2_dist':>10}")
print("-" * 75)

for algo in ["dtw", "ddtw", "fastdtw"]:
    for metric in ["euclidean", "cosine"]:
        comp = ActionComparator(algorithm=algo, metric=metric, similarity_sigma=1.0)
        rs = comp.compare(tpl, tpl, "self")
        r1 = comp.compare(t1, tpl, "t1")
        r2 = comp.compare(t2, tpl, "t2")
        print(f"{algo:<12} {metric:<12} {rs.similarity:>7.2%} {r1.similarity:>7.2%} {r2.similarity:>7.2%} {r1.distance:>10.2f} {r2.distance:>10.2f}")

estimator.close()
