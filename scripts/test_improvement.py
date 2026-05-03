"""
测试改进效果：对比改进前后的相似度评分

用法:
    python scripts/test_improvement.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.video_source import FileSource
from src.action_comparison.comparison import ActionComparator

# 视频路径
TEMPLATE_VIDEO = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_0.avi"
TEST_VIDEO_1 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_1.avi"
TEST_VIDEO_2 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_2.avi"


def extract_sequence(estimator, video_path):
    """从视频提取骨骼序列"""
    print(f"\n提取骨骼: {os.path.basename(video_path)}")
    with FileSource(video_path) as source:
        seq = estimator.estimate_video(source, max_frames=300, show_progress=False)
    print(f"  帧数: {seq.num_frames}")
    return seq


def test_comparison():
    print("=" * 70)
    print("改进效果测试")
    print("=" * 70)

    # 初始化
    estimator = PoseEstimator()

    # 提取骨骼
    template_seq = extract_sequence(estimator, TEMPLATE_VIDEO)
    test1_seq = extract_sequence(estimator, TEST_VIDEO_1)
    test2_seq = extract_sequence(estimator, TEST_VIDEO_2)

    # 测试改进后的对比（DDTW，与 app 的 auto 模式一致，sigma=1.0）
    print("\n" + "=" * 70)
    print("改进后对比结果（DDTW + 片段提取 + 骨骼过滤 + 平移归一化 + 高斯相似度）")
    print("=" * 70)

    comparator = ActionComparator(
        algorithm="ddtw",
        metric="euclidean",
        preprocess=True,
        target_frames=60,
        similarity_sigma=1.0,
    )

    # 模板自对比
    result_self = comparator.compare(template_seq, template_seq, "self")
    print(f"\n模板自对比:")
    print(f"  距离: {result_self.distance:.4f}")
    print(f"  相似度: {result_self.similarity:.2%}")
    print(f"  路径长度: {result_self.path_length}")

    # 测试1
    result1 = comparator.compare(test1_seq, template_seq, "test1")
    print(f"\n测试视频1 (有遮挡):")
    print(f"  距离: {result1.distance:.4f}")
    print(f"  相似度: {result1.similarity:.2%}")
    print(f"  路径长度: {result1.path_length}")

    # 测试2
    result2 = comparator.compare(test2_seq, template_seq, "test2")
    print(f"\n测试视频2 (有准备动作):")
    print(f"  距离: {result2.distance:.4f}")
    print(f"  相似度: {result2.similarity:.2%}")
    print(f"  路径长度: {result2.path_length}")

    # 测试不同 sigma 值的效果
    print("\n" + "=" * 70)
    print("sigma 参数灵敏度分析")
    print("=" * 70)

    for sigma in [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]:
        comp = ActionComparator(
            algorithm="ddtw", metric="euclidean",
            preprocess=True, target_frames=60,
            similarity_sigma=sigma,
        )
        r_self = comp.compare(template_seq, template_seq, "self")
        r1 = comp.compare(test1_seq, template_seq, "test1")
        r2 = comp.compare(test2_seq, template_seq, "test2")
        print(f"  sigma={sigma:.1f}: 自对比={r_self.similarity:.2%}, "
              f"测试1={r1.similarity:.2%}, 测试2={r2.similarity:.2%}")

    # 质量评分预估（使用 sigma=1.0 的结果）
    # 重新用 sigma=1.0 跑一次
    comparator_final = ActionComparator(
        algorithm="ddtw", metric="euclidean",
        preprocess=True, target_frames=60,
        similarity_sigma=1.0,
    )
    r_self_f = comparator_final.compare(template_seq, template_seq, "self")
    r1_f = comparator_final.compare(test1_seq, template_seq, "test1")
    r2_f = comparator_final.compare(test2_seq, template_seq, "test2")

    from src.correction.feedback import FeedbackGenerator
    print("\n" + "-" * 50)
    print("最终结果 (sigma=1.0):")
    for name, sim in [("自对比", r_self_f.similarity),
                       ("测试1", r1_f.similarity),
                       ("测试2", r2_f.similarity)]:
        if sim >= 0.999:
            score = 100.0
        elif sim >= 0.85:
            score = 90.0 + (sim - 0.85) / 0.15 * 10.0
        elif sim >= 0.65:
            score = 65.0 + (sim - 0.65) / 0.2 * 25.0
        elif sim >= 0.4:
            score = 35.0 + (sim - 0.4) / 0.25 * 30.0
        elif sim >= 0.15:
            score = 10.0 + (sim - 0.15) / 0.25 * 25.0
        else:
            score = sim / 0.15 * 10.0
        print(f"  {name}: 相似度={sim:.2%} → 评分={score:.1f}")

    print("\n" + "=" * 70)
    print("测试完成!")

    estimator.close()


if __name__ == "__main__":
    test_comparison()
