"""
诊断脚本：模拟两种场景，追踪评分流水线的每个环节。

场景1: 垂直举手 + 交替摆小臂  vs  水平举手 + 交替摆小臂
场景2: 垂直举手 + 交替摆小臂  vs  垂直举手 + 同步摆小臂
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

from src.pose_estimation.data_types import PoseSequence, PoseFrame, PoseLandmark
from src.action_comparison.comparison import ActionComparator
from src.action_comparison.distance_metrics import (
    sequence_to_feature_matrix,
    sequence_to_hybrid_matrix,
    sequence_to_landmark_matrix,
)
from src.correction.angle_utils import AngleCalculator, ANGLE_DEFINITIONS

# ============================================================
# 工具函数：生成模拟关键点
# ============================================================

def make_landmark(x, y, z=0.0, vis=1.0):
    """创建单个关键点"""
    return PoseLandmark(x=x, y=y, z=z, visibility=vis)


def make_standing_pose(shoulder_elevation_deg, left_elbow_deg, right_elbow_deg):
    """
    生成一站姿帧。
    
    shoulder_elevation_deg: 上臂相对于躯干的角度
      0° = 手臂竖直上举（贴在耳边）
      90° = 手臂水平前举
      180° = 手臂自然下垂
    
    left_elbow_deg: 左肘弯曲角 (180°=伸直, 90°=弯曲)
    right_elbow_deg: 右肘弯曲角
    """
    # 使用简化的 2D 坐标系（y 轴向上为正，MediaPipe 是向下为正，这里统一处理）
    # 髋中点作为原点
    # 躯干长度 ≈ 1.0（归一化单位）
    
    # 关键点坐标（髋中心归一化坐标系）
    # 这里用简单位置模拟，确保角度计算合理
    
    import math
    
    shoulder_rad = math.radians(shoulder_elevation_deg)
    left_elbow_rad = math.radians(left_elbow_deg)
    right_elbow_rad = math.radians(right_elbow_deg)
    
    # 躯干
    hip_y = 0.5       # 髋在原点下方
    shoulder_y = -0.3  # 肩在原点上方
    shoulder_x_offset = 0.2  # 肩宽的一半
    
    # 头部（鼻子）
    nose_x, nose_y, nose_z = 0.0, -0.65, 0.0
    
    # 左肩、右肩
    l_shoulder = (-shoulder_x_offset, shoulder_y, 0.0)
    r_shoulder = (shoulder_x_offset, shoulder_y, 0.0)
    
    # 上臂方向：shoulder_elevation 是上臂与竖直向下方向的夹角
    # shoulder_elevation=0 → 竖直向上（手臂贴耳）
    # shoulder_elevation=90 → 水平向前
    # shoulder_elevation=180 → 竖直向下（自然下垂）
    
    upper_arm_len = 0.4
    
    # 上臂方向向量（从肩出发）
    # 简化：在 x-y 平面内
    upper_dir_x = math.sin(shoulder_rad)  # 水平分量
    upper_dir_y = -math.cos(shoulder_rad)  # 垂直分量（向上为负）
    
    l_elbow = (
        l_shoulder[0] + upper_dir_x * upper_arm_len,
        l_shoulder[1] + upper_dir_y * upper_arm_len,
        0.0
    )
    r_elbow = (
        r_shoulder[0] + upper_dir_x * upper_arm_len,
        r_shoulder[1] + upper_dir_y * upper_arm_len,
        0.0
    )
    
    # 前臂方向：从肘出发，肘角=180°时继续沿上臂方向
    forearm_len = 0.35
    
    # 左前臂：肘角相对于上臂的偏转
    # elbow_deg=180 → 完全伸直（与上臂同向）
    # elbow_deg=90 → 弯曲90°
    def forearm_end(elbow, upper_dir, elbow_deg, side_sign):
        er = math.radians(elbow_deg)
        # 前臂方向：从上臂方向继续偏转
        # 简化：前臂在 x-y 平面内弯曲
        fx = upper_dir[0] * math.cos(math.pi - er) + upper_dir[1] * math.sin(math.pi - er) * side_sign
        fy = -upper_dir[0] * math.sin(math.pi - er) * side_sign + upper_dir[1] * math.cos(math.pi - er)
        return (
            elbow[0] + fx * forearm_len,
            elbow[1] + fy * forearm_len,
            0.0
        )
    
    # 上臂单位方向
    upper_dir = (upper_dir_x, upper_dir_y)
    
    l_wrist = forearm_end(l_elbow, upper_dir, left_elbow_deg, -1)
    r_wrist = forearm_end(r_elbow, upper_dir, right_elbow_deg, 1)
    
    # 下肢（站立姿态）
    l_hip = (-0.12, hip_y, 0.0)
    r_hip = (0.12, hip_y, 0.0)
    l_knee = (-0.12, hip_y + 0.55, 0.0)
    r_knee = (0.12, hip_y + 0.55, 0.0)
    l_ankle = (-0.12, hip_y + 1.0, 0.0)
    r_ankle = (0.12, hip_y + 1.0, 0.0)
    l_heel = (-0.12, hip_y + 1.05, 0.0)
    r_heel = (0.12, hip_y + 1.05, 0.0)
    l_foot = (-0.12, hip_y + 1.1, 0.0)
    r_foot = (0.12, hip_y + 1.1, 0.0)
    
    # 构建所有33个关键点（MediaPipe格式）
    # 索引参照 MediaPipe 标准
    pts = {}
    pts[0] = (nose_x, nose_y, nose_z, 1.0)        # nose
    pts[1] = (0.04, -0.62, 0.0, 1.0)               # left_eye_inner
    pts[2] = (0.02, -0.62, 0.0, 1.0)               # left_eye
    pts[3] = (0.0, -0.62, 0.0, 1.0)                # left_eye_outer
    pts[4] = (-0.04, -0.62, 0.0, 1.0)              # right_eye_inner
    pts[5] = (-0.02, -0.62, 0.0, 1.0)              # right_eye
    pts[6] = (0.0, -0.62, 0.0, 1.0)                # right_eye_outer
    pts[7] = (0.08, -0.60, 0.0, 1.0)               # left_ear
    pts[8] = (-0.08, -0.60, 0.0, 1.0)              # right_ear
    pts[9] = (0.03, -0.55, 0.0, 1.0)               # mouth_left
    pts[10] = (-0.03, -0.55, 0.0, 1.0)             # mouth_right
    
    pts[11] = (*l_shoulder, 1.0)                    # left_shoulder
    pts[12] = (*r_shoulder, 1.0)                    # right_shoulder
    pts[13] = (*l_elbow, 1.0)                       # left_elbow
    pts[14] = (*r_elbow, 1.0)                       # right_elbow
    pts[15] = (*l_wrist, 1.0)                       # left_wrist
    pts[16] = (*r_wrist, 1.0)                       # right_wrist
    pts[17] = (0.0, -0.67, 0.0, 1.0)               # left_pinky
    pts[18] = (0.0, -0.67, 0.0, 1.0)               # right_pinky
    pts[19] = (0.0, -0.67, 0.0, 1.0)               # left_index
    pts[20] = (0.0, -0.67, 0.0, 1.0)               # right_index
    pts[21] = (0.0, -0.67, 0.0, 1.0)               # left_thumb
    pts[22] = (0.0, -0.67, 0.0, 1.0)               # right_thumb
    
    pts[23] = (*l_hip, 1.0)                         # left_hip
    pts[24] = (*r_hip, 1.0)                         # right_hip
    pts[25] = (*l_knee, 1.0)                        # left_knee
    pts[26] = (*r_knee, 1.0)                        # right_knee
    pts[27] = (*l_ankle, 1.0)                       # left_ankle
    pts[28] = (*r_ankle, 1.0)                       # right_ankle
    pts[29] = (*l_heel, 1.0)                        # left_heel
    pts[30] = (*r_heel, 1.0)                        # right_heel
    pts[31] = (*l_foot, 1.0)                        # left_foot_index
    pts[32] = (*r_foot, 1.0)                        # right_foot_index
    
    # 构建 PoseLandmark 列表
    landmarks = []
    for i in range(33):
        x, y, z, v = pts[i]
        landmarks.append(make_landmark(x, y, z, v))
    
    return landmarks


def create_sequence_from_poses(poses_landmarks):
    """从多个帧的关键点列表创建 PoseSequence"""
    frames = []
    for i, lm_list in enumerate(poses_landmarks):
        frames.append(PoseFrame(landmarks=lm_list, timestamp=0.0, frame_index=i))
    return PoseSequence(frames=frames)


def generate_scenario1():
    """场景1：垂直举手(shoulder≈180°) vs 水平举手(shoulder≈90°)，均交替摆小臂"""
    seq_a_frames = []
    seq_b_frames = []
    
    # 模拟 60 帧，小臂以正弦波交替摆动
    for t in range(60):
        phase = t / 60.0 * 4 * np.pi  # 2 个完整周期
        left_elbow = 90 + 45 * np.sin(phase)          # 左肘在 45°-135° 间摆动
        right_elbow = 90 + 45 * np.sin(phase + np.pi)  # 右肘反相
        
        # 序列 A: 垂直举手 (shoulder ≈ 180°)
        seq_a_frames.append(make_standing_pose(180, left_elbow, right_elbow))
        # 序列 B: 水平举手 (shoulder ≈ 90°)
        seq_b_frames.append(make_standing_pose(90, left_elbow, right_elbow))
    
    return (create_sequence_from_poses(seq_a_frames),
            create_sequence_from_poses(seq_b_frames))


def generate_scenario2():
    """场景2：垂直举手 + 交替摆小臂 vs 垂直举手 + 同步摆小臂"""
    seq_a_frames = []
    seq_b_frames = []
    
    for t in range(60):
        phase = t / 60.0 * 4 * np.pi
        
        # 序列 A: 交替 (反相)
        left_a = 90 + 45 * np.sin(phase)
        right_a = 90 + 45 * np.sin(phase + np.pi)
        seq_a_frames.append(make_standing_pose(180, left_a, right_a))
        
        # 序列 B: 同步 (同相)
        left_b = 90 + 45 * np.sin(phase)
        right_b = 90 + 45 * np.sin(phase)  # 与左肘同相
        seq_b_frames.append(make_standing_pose(180, left_b, right_b))
    
    return (create_sequence_from_poses(seq_a_frames),
            create_sequence_from_poses(seq_b_frames))


# ============================================================
# 诊断分析
# ============================================================

def diagnose(scenario_name, seq_a, seq_b):
    print(f"\n{'='*70}")
    print(f"  {scenario_name}")
    print(f"{'='*70}")
    
    # 1. 检查原始角度
    calc = AngleCalculator()
    arr_a = seq_a.to_numpy()
    arr_b = seq_b.to_numpy()
    
    print(f"\n  序列 A: {seq_a.num_frames} 帧, 序列 B: {seq_b.num_frames} 帧")
    
    # 取中间帧查看角度
    mid = seq_a.num_frames // 2
    angles_a = calc.compute_frame_angles(arr_a[mid])
    angles_b = calc.compute_frame_angles(arr_b[mid])
    
    print(f"\n  --- 中间帧角度对比 ---")
    print(f"  {'角度':<25} {'序列A':>8} {'序列B':>8} {'差值':>8}")
    print(f"  {'-'*49}")
    for name in ANGLE_DEFINITIONS:
        a = angles_a[name]
        b = angles_b[name]
        diff = a - b
        marker = " ← 关键差异" if abs(diff) > 10 else ""
        print(f"  {name:<25} {a:>7.1f}° {b:>7.1f}° {diff:>+7.1f}°{marker}")
    
    # 2. 特征矩阵分析
    print(f"\n  --- 特征矩阵（angle-only, 默认 euclidean 模式）---")
    feat_a = sequence_to_feature_matrix(seq_a)
    feat_b = sequence_to_feature_matrix(seq_b)
    print(f"  维度: {feat_a.shape[1]} (即 {len(ANGLE_DEFINITIONS)} 个角度)")
    
    # 逐帧距离统计
    min_len = min(feat_a.shape[0], feat_b.shape[0])
    frame_dists = []
    for i in range(min_len):
        d = np.sqrt(np.sum((feat_a[i] - feat_b[i]) ** 2))
        frame_dists.append(d)
    
    frame_dists = np.array(frame_dists)
    print(f"  逐帧(直接对齐) L2 距离: mean={frame_dists.mean():.4f}, "
          f"min={frame_dists.min():.4f}, max={frame_dists.max():.4f}")
    
    # 相似度（如果直接逐帧对齐）
    sigma = 0.7
    raw_sim_direct = float(np.exp(-(frame_dists.mean() ** 2) / (2 * sigma ** 2)))
    print(f"  直接对齐相似度(sigma=0.7): {raw_sim_direct:.2%}")
    
    # 3. 混合特征矩阵分析
    print(f"\n  --- 混合特征矩阵（hybrid, 坐标+角度）---")
    hybrid_a = sequence_to_hybrid_matrix(seq_a)
    hybrid_b = sequence_to_hybrid_matrix(seq_b)
    print(f"  维度: {hybrid_a.shape[1]} (坐标51维 + 角度9维)")
    
    hybrid_dists = []
    for i in range(min_len):
        d = np.sqrt(np.sum((hybrid_a[i] - hybrid_b[i]) ** 2))
        hybrid_dists.append(d)
    hybrid_dists = np.array(hybrid_dists)
    print(f"  逐帧 L2 距离: mean={hybrid_dists.mean():.4f}, "
          f"min={hybrid_dists.min():.4f}, max={hybrid_dists.max():.4f}")
    raw_sim_hybrid = float(np.exp(-(hybrid_dists.mean() ** 2) / (2 * sigma ** 2)))
    print(f"  直接对齐相似度(sigma=0.7): {raw_sim_hybrid:.2%}")
    
    # 4. 实际 DTW 对比（默认 euclidean = angle-only）
    print(f"\n  --- ActionComparator 对比（默认 metric=euclidean, sigma=0.7）---")
    comp = ActionComparator(algorithm="dtw", metric="euclidean",
                            preprocess=True, target_frames=60,
                            similarity_sigma=0.7)
    result = comp.compare(seq_a, seq_b, template_name="test")
    print(f"  DTW 距离: {result.distance:.4f}")
    print(f"  路径长度: {result.path_length}")
    print(f"  归一化距离: {result.normalized_distance:.4f}")
    print(f"  原始相似度: {result.raw_similarity:.2%}")
    print(f"  覆盖度: {result.template_coverage:.2%}")
    print(f"  覆盖因子: {result.coverage_factor:.2%}")
    print(f"  最终相似度: {result.similarity:.2%}")
    
    # 5. 使用 hybrid metric 对比
    print(f"\n  --- ActionComparator 对比（metric=hybrid, sigma=0.7）---")
    comp_h = ActionComparator(algorithm="dtw", metric="hybrid",
                              preprocess=True, target_frames=60,
                              similarity_sigma=0.7)
    result_h = comp_h.compare(seq_a, seq_b, template_name="test")
    print(f"  DTW 距离: {result_h.distance:.4f}")
    print(f"  路径长度: {result_h.path_length}")
    print(f"  归一化距离: {result_h.normalized_distance:.4f}")
    print(f"  原始相似度: {result_h.raw_similarity:.2%}")
    print(f"  覆盖度: {result_h.template_coverage:.2%}")
    print(f"  覆盖因子: {result_h.coverage_factor:.2%}")
    print(f"  最终相似度: {result_h.similarity:.2%}")
    
    # 6. 不同 sigma 的敏感性测试
    print(f"\n  --- 不同 sigma 的相似度 (angle-only) ---")
    for s in [0.3, 0.5, 0.7, 1.0]:
        comp_s = ActionComparator(algorithm="dtw", metric="euclidean",
                                  preprocess=True, target_frames=60,
                                  similarity_sigma=s)
        result_s = comp_s.compare(seq_a, seq_b, template_name="test")
        print(f"  sigma={s:.1f}: raw_sim={result_s.raw_similarity:.2%}, "
              f"final={result_s.similarity:.2%}")


if __name__ == "__main__":
    print("=" * 70)
    print("  评分区分度诊断")
    print("=" * 70)
    
    # 场景1
    seq_a1, seq_b1 = generate_scenario1()
    diagnose("场景1: 垂直举手 vs 水平举手（均交替摆小臂）", seq_a1, seq_b1)
    
    # 场景2
    seq_a2, seq_b2 = generate_scenario2()
    diagnose("场景2: 交替摆小臂 vs 同步摆小臂（均垂直举手）", seq_a2, seq_b2)
    
    print(f"\n{'='*70}")
    print("  诊断完成")
    print(f"{'='*70}")
