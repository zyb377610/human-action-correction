"""最终效果测试：使用优化后的 auto 模式 (dtw + cosine)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.video_source import FileSource
from src.action_comparison.comparison import ActionComparator

TEMPLATE = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_0.avi"
TEST1 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_1.avi"
TEST2 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_2.avi"

print("=" * 70)
print("最终优化效果测试 (auto 模式: dtw + cosine)")
print("=" * 70)

estimator = PoseEstimator()
with FileSource(TEMPLATE) as s: tpl = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST1) as s: t1 = estimator.estimate_video(s, show_progress=False)
with FileSource(TEST2) as s: t2 = estimator.estimate_video(s, show_progress=False)

# 使用 auto 模式的配置: dtw + cosine
comp = ActionComparator(algorithm="dtw", metric="cosine", similarity_sigma=1.0)

rs = comp.compare(tpl, tpl, "self")
r1 = comp.compare(t1, tpl, "test1")
r2 = comp.compare(t2, tpl, "test2")

print(f"\n模板自对比: 相似度={rs.similarity:.2%}, 距离={rs.distance:.4f}")
print(f"测试1 (有遮挡): 相似度={r1.similarity:.2%}, 距离={r1.distance:.4f}")
print(f"测试2 (有准备动作): 相似度={r2.similarity:.2%}, 距离={r2.distance:.4f}")

# 验证裁剪信息是否正确存储
print(f"\n裁剪信息:")
print(f"  测试1: query_crop={r1.query_crop_range}, tpl_crop={r1.template_crop_range}")
print(f"  测试2: query_crop={r2.query_crop_range}, tpl_crop={r2.template_crop_range}")

# 质量评分
print(f"\n质量评分:")
for name, sim in [("自对比", rs.similarity), ("测试1", r1.similarity), ("测试2", r2.similarity)]:
    if sim >= 0.999: score = 100.0
    elif sim >= 0.85: score = 90.0 + (sim - 0.85) / 0.15 * 10.0
    elif sim >= 0.65: score = 65.0 + (sim - 0.65) / 0.2 * 25.0
    elif sim >= 0.4: score = 35.0 + (sim - 0.4) / 0.25 * 30.0
    elif sim >= 0.15: score = 10.0 + (sim - 0.15) / 0.25 * 25.0
    else: score = sim / 0.15 * 10.0
    print(f"  {name}: {sim:.2%} → {score:.1f}分")

print(f"\n{'=' * 70}")
print("对比（改进前 → 改进后）:")
print(f"  测试1: 45.28% / 43.0分 → {r1.similarity:.2%} / {90.0 + (r1.similarity - 0.85) / 0.15 * 10.0 if r1.similarity >= 0.85 else 65.0 + (r1.similarity - 0.65) / 0.2 * 25.0:.1f}分")
print(f"  测试2: 49.15% / 48.7分 → {r2.similarity:.2%} / {90.0 + (r2.similarity - 0.85) / 0.15 * 10.0 if r2.similarity >= 0.85 else 65.0 + (r2.similarity - 0.65) / 0.2 * 25.0:.1f}分")
print("=" * 70)

estimator.close()
