"""验证：不同动作之间的区分度"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.WARNING)

from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.video_source import FileSource
from src.action_comparison.comparison import ActionComparator

TEMPLATE = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_0.avi"
TEST1 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_1.avi"

# 找一个完全不同的动作视频
import glob
other_dirs = glob.glob(r"E:\毕业设计\test-resource\hmdb51_org\*")
other_videos = []
for d in other_dirs:
    if "throw" not in d:
        vids = glob.glob(os.path.join(d, "*.avi"))
        if vids:
            other_videos.append(vids[0])
    if len(other_videos) >= 3:
        break

estimator = PoseEstimator()
with FileSource(TEMPLATE) as s: tpl = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST1) as s: t1 = estimator.estimate_video(s, show_progress=False)

other_seqs = []
for v in other_videos:
    try:
        with FileSource(v) as s:
            seq = estimator.estimate_video(s, show_progress=False)
        if seq.num_frames > 10:
            other_seqs.append((os.path.basename(os.path.dirname(v)), seq))
    except:
        pass

print(f"{'Algo':<10} {'Metric':<10} {'Same(T1)':>10} {'Diff1':>10} {'Diff2':>10} {'Diff3':>10}  | 能区分?")
print("-" * 85)

for algo, metric in [("dtw","cosine"), ("ddtw","euclidean"), ("ddtw","cosine"), ("dtw","euclidean")]:
    comp = ActionComparator(algorithm=algo, metric=metric, similarity_sigma=1.0)
    r1 = comp.compare(t1, tpl, "same")
    diffs = []
    diff_names = []
    for name, seq in other_seqs[:3]:
        r = comp.compare(seq, tpl, name)
        diffs.append(r.similarity)
        diff_names.append(name)

    same_score = r1.similarity
    can_distinguish = all(d < same_score * 0.7 for d in diffs)
    diff_strs = [f"{d:>9.2%}" for d in diffs]
    pad = ["" ] * (3 - len(diff_strs))
    diff_strs += [f"{'N/A':>9}"] * (3 - len(diff_strs))

    print(f"{algo:<10} {metric:<10} {same_score:>9.2%} {diff_strs[0]} {diff_strs[1]} {diff_strs[2]}  | {'YES' if can_distinguish else 'NO!!!'}")

estimator.close()
