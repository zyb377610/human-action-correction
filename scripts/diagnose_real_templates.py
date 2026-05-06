"""
使用用户真实模板数据验证评分区分度问题。
对比三组模板:
  test_shake1: 双手举起交替上下
  test_shake2: 双手举起同时上下
  test_shake3: 双手举平交替上下
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

from src.pose_estimation.data_types import PoseSequence
from src.action_comparison.comparison import ActionComparator
from src.action_comparison.distance_metrics import (
    sequence_to_feature_matrix,
    sequence_to_hybrid_matrix,
    compute_valid_joints,
    CORE_JOINT_INDICES,
    CORE_JOINT_NAMES,
)
from src.correction.angle_utils import AngleCalculator, ANGLE_DEFINITIONS

BASE = r"E:\毕业设计\human-action-correction\data\templates"

TEMPLATES = {
    "shake1_交替_上": os.path.join(BASE, "test_shake1", "cam_template_1778052252.json"),
    "shake2_同步_上": os.path.join(BASE, "test_shake2", "cam_template_1778052271.json"),
    "shake3_交替_平": os.path.join(BASE, "test_shake3", "cam_template_1778052289.json"),
}

def load_seq(path):
    seq = PoseSequence.load(path)
    print(f"  加载: {os.path.basename(path)} → {seq.num_frames} 帧")
    return seq

def check_visibility(seq, label):
    """检查序列中各关节的可见度情况"""
    valid = compute_valid_joints(seq)
    excluded = [CORE_JOINT_NAMES.get(j, str(j)) for j in CORE_JOINT_INDICES if j not in valid]
    print(f"  [{label}] 有效关节: {len(valid)}/17, 排除: {excluded if excluded else '无'}")
    
    # 额外：检查角度相关关节的可见度
    from src.correction.angle_utils import compute_valid_angles, ANGLE_DEFINITIONS as _AD
    arr = seq.to_numpy()
    T = arr.shape[0]
    print(f"  [{label}] 下半身关键点平均可见度:")
    for j in [23, 24, 25, 26, 27, 28]:
        name = CORE_JOINT_NAMES.get(j, str(j))
        avg_vis = float(np.mean(arr[:, j, 3]))
        above = float(np.mean(arr[:, j, 3] >= 0.3))
        print(f"    {name}({j}): 平均vis={avg_vis:.4f}, ≥0.3占比={above:.1%}")
    
    # 展示两种模式的有效角度
    valid_angles_strict = compute_valid_angles(seq)  # 回退模式（三点可见度）
    valid_angles_vertex = compute_valid_angles(seq, valid_joint_indices=valid)  # 顶点模式
    print(f"  [{label}] 有效角度(三点模式): {len(valid_angles_strict)}/9 = {valid_angles_strict}")
    print(f"  [{label}] 有效角度(顶点模式): {len(valid_angles_vertex)}/9 = {valid_angles_vertex}")

def show_angle_summary(seq, label):
    """展示序列的平均角度"""
    calc = AngleCalculator()
    avg_angles = calc.compute_sequence_angles(seq)
    print(f"  [{label}] 平均角度:")
    for name in ANGLE_DEFINITIONS:
        print(f"    {name:<25}: {avg_angles[name]:>6.1f}°")

def run_comparison(seq_a, label_a, seq_b, label_b):
    """对比两个序列，输出多种模式的结果"""
    print(f"\n{'='*65}")
    print(f"  对比: {label_a}  vs  {label_b}")
    print(f"{'='*65}")
    
    # 角度差异概览
    calc = AngleCalculator()
    avg_a = calc.compute_sequence_angles(seq_a)
    avg_b = calc.compute_sequence_angles(seq_b)
    print(f"\n  --- 平均角度差异 ---")
    diffs = {}
    for name in ANGLE_DEFINITIONS:
        d = abs(avg_a[name] - avg_b[name])
        diffs[name] = d
        marker = " ←" if d > 10 else ""
        print(f"  {name:<25}: {avg_a[name]:>6.1f}° vs {avg_b[name]:>6.1f}°  Δ={d:>5.1f}°{marker}")
    print(f"  显著差异角度数: {sum(1 for d in diffs.values() if d > 10)} / 9")
    
    # 不同 metric 对比
    configs = [
        ("euclidean (angle-only, 默认)", "euclidean", 0.7),
        ("hybrid (坐标+角度)", "hybrid", 0.7),
        ("euclidean + sigma=0.4", "euclidean", 0.4),
    ]
    
    for desc, metric, sigma in configs:
        comp = ActionComparator(
            algorithm="dtw", metric=metric,
            preprocess=True, target_frames=60,
            similarity_sigma=sigma,
        )
        result = comp.compare(seq_a, seq_b, template_name=label_b)
        status = "✅ 合理" if result.similarity < 0.80 else ("⚠️ 偏高" if result.similarity < 0.90 else "❌ 过高")
        print(f"\n  [{desc}] sigma={sigma}")
        print(f"    DTW距离={result.distance:.3f}  路径长={result.path_length}  "
              f"归一化距离={result.normalized_distance:.4f}")
        print(f"    原始相似度={result.raw_similarity:.2%}  "
              f"覆盖度={result.template_coverage:.2%}  "
              f"覆盖因子={result.coverage_factor:.2%}")
        print(f"    最终相似度={result.similarity:.2%}  {status}")


if __name__ == "__main__":
    print("=" * 65)
    print("  真实模板评分区分度验证")
    print("=" * 65)
    
    # 加载
    print("\n--- 加载模板 ---")
    seqs = {}
    for name, path in TEMPLATES.items():
        seqs[name] = load_seq(path)
    
    # 可见度检查
    print("\n--- 可见度剔除检查 ---")
    for name, seq in seqs.items():
        check_visibility(seq, name)
    
    # 角度摘要
    print("\n--- 平均角度摘要 ---")
    for name, seq in seqs.items():
        show_angle_summary(seq, name)
    
    # 场景1: shake1(交替上) vs shake3(交替平) — 只改肩角
    run_comparison(seqs["shake1_交替_上"], "双手上举+交替", 
                   seqs["shake3_交替_平"], "双手平举+交替")
    
    # 场景2: shake1(交替上) vs shake2(同步上) — 只改肘角相位
    run_comparison(seqs["shake1_交替_上"], "双手上举+交替",
                   seqs["shake2_同步_上"], "双手上举+同步")
    
    print("\n" + "=" * 65)
    print("  验证完成")
    print("=" * 65)
