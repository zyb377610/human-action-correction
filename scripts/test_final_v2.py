"""最终验证：角度特征 + sigma=0.7"""
import sys, os, glob
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
print("Loading...", flush=True)
with FileSource(TEMPLATE) as s: tpl = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST1) as s: t1 = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST2) as s: t2 = estimator.estimate_video(s, show_progress=False)

# 找异类动作
diffs = []
for d in sorted(glob.glob(r"E:\毕业设计\test-resource\hmdb51_org\*")):
    nm = os.path.basename(d)
    if "throw" not in nm:
        vids = glob.glob(os.path.join(d, "*.avi"))
        if vids:
            try:
                with FileSource(vids[0]) as s:
                    seq = estimator.estimate_video(s, show_progress=False)
                if seq.num_frames > 10:
                    diffs.append((nm, seq))
            except: pass
    if len(diffs) >= 2: break

comp = ActionComparator(algorithm="dtw", metric="euclidean", similarity_sigma=0.7)

print("\nComparing...", flush=True)
rs = comp.compare(tpl, tpl, "self")
r1 = comp.compare(t1, tpl, "t1")
r2 = comp.compare(t2, tpl, "t2")

def to_score(sim):
    if sim >= 0.999: return 100.0
    elif sim >= 0.85: return 90.0 + (sim - 0.85) / 0.15 * 10.0
    elif sim >= 0.70: return 70.0 + (sim - 0.70) / 0.15 * 20.0
    elif sim >= 0.50: return 45.0 + (sim - 0.50) / 0.20 * 25.0
    elif sim >= 0.30: return 20.0 + (sim - 0.30) / 0.20 * 25.0
    else: return sim / 0.30 * 20.0

print(f"\n{'='*60}")
print(f"最终结果 (角度特征 + DTW + sigma=0.7)")
print(f"{'='*60}")
print(f"  模板自对比:          相似度={rs.similarity:.2%}  评分={to_score(rs.similarity):.1f}")
print(f"  测试1(同类,有遮挡):   相似度={r1.similarity:.2%}  评分={to_score(r1.similarity):.1f}")
print(f"  测试2(同类,有准备):   相似度={r2.similarity:.2%}  评分={to_score(r2.similarity):.1f}")

for nm, seq in diffs:
    rd = comp.compare(seq, tpl, nm)
    print(f"  {nm:20s}(异类): 相似度={rd.similarity:.2%}  评分={to_score(rd.similarity):.1f}")

print(f"\n对比改进前 → 改进后:")
print(f"  测试1: 45.28%/43分 → {r1.similarity:.2%}/{to_score(r1.similarity):.1f}分")
print(f"  测试2: 49.15%/49分 → {r2.similarity:.2%}/{to_score(r2.similarity):.1f}分")

estimator.close()
print("Done!", flush=True)
