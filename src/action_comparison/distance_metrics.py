"""
距离度量函数

支持欧氏距离、余弦距离、曼哈顿距离，提供统一接口。
"""

from typing import Callable, Optional, Sequence

import numpy as np


# ===== 核心关节点索引 =====
# 从 MediaPipe 33 个关键点中筛选 17 个关键骨骼点用于 DTW 比对。
# 包含：头部参考点(1) + 上肢(6) + 躯干(4) + 下肢(6) = 17 个
# 去掉眼睛(1-4)、耳朵(7-8)、嘴角(9-10)、小指/拇指(19-22)等
# 高抖动低信息量的点，保留运动分析核心骨骼。
CORE_JOINT_INDICES = [
    0,   # nose          — 头部朝向参考
    11,  # left_shoulder  — 左肩
    12,  # right_shoulder — 右肩
    13,  # left_elbow     — 左肘
    14,  # right_elbow    — 右肘
    15,  # left_wrist     — 左腕
    16,  # right_wrist    — 右腕
    23,  # left_hip       — 左髋
    24,  # right_hip      — 右髋
    25,  # left_knee      — 左膝
    26,  # right_knee     — 右膝
    27,  # left_ankle     — 左踝
    28,  # right_ankle    — 右踝
    29,  # left_heel      — 左脚跟
    30,  # right_heel     — 右脚跟
    31,  # left_foot_index  — 左脚尖
    32,  # right_foot_index — 右脚尖
]

# 核心关节点中文名称
CORE_JOINT_NAMES = {
    0: "鼻子",
    11: "左肩", 12: "右肩",
    13: "左肘", 14: "右肘",
    15: "左腕", 16: "右腕",
    23: "左髋", 24: "右髋",
    25: "左膝", 26: "右膝",
    27: "左踝", 28: "右踝",
    29: "左脚跟", 30: "右脚跟",
    31: "左脚尖", 32: "右脚尖",
}


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦距离 = 1 - cosine_similarity

    范围 [0, 2]，0 表示方向完全一致
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0  # 零向量视为最大不相似
    return float(1.0 - dot / (norm_a * norm_b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """曼哈顿距离（L1 范数）"""
    return float(np.sum(np.abs(a - b)))


# ===== 统一接口 =====

_METRICS = {
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "manhattan": manhattan_distance,
}


def get_distance_func(name: str) -> Callable:
    """
    获取距离度量函数

    Args:
        name: 度量名称 — "euclidean" / "cosine" / "manhattan"

    Returns:
        距离函数 (a, b) -> float
    """
    name = name.lower().strip()
    if name not in _METRICS:
        raise ValueError(
            f"不支持的距离度量: '{name}'，可选: {list(_METRICS.keys())}"
        )
    return _METRICS[name]


def sequence_to_feature_matrix(
    sequence,
    joint_indices: Optional[Sequence[int]] = None,
    normalize_body_scale: bool = True,
) -> np.ndarray:
    """
    将 PoseSequence 转换为特征矩阵

    Args:
        sequence: PoseSequence 对象
        joint_indices: 要使用的关节点索引，None 时用 CORE_JOINT_INDICES
        normalize_body_scale: 是否按躯干长度归一化（消除体型差异）

    Returns:
        (T, num_joints * 3) ndarray
    """
    if joint_indices is None:
        joint_indices = CORE_JOINT_INDICES

    T = sequence.num_frames
    num_joints = len(joint_indices)
    matrix = np.zeros((T, num_joints * 3), dtype=np.float64)

    # 计算身体尺度因子（躯干长度：肩中点到髋中点）
    scale_factor = 1.0
    if normalize_body_scale:
        scale_factor = _compute_body_scale(sequence)
        if scale_factor < 0.05:
            scale_factor = 1.0  # 兜底

    for t, frame in enumerate(sequence.frames):
        for k, j in enumerate(joint_indices):
            if j < len(frame.landmarks):
                lm = frame.landmarks[j]
                matrix[t, k * 3] = lm.x / scale_factor
                matrix[t, k * 3 + 1] = lm.y / scale_factor
                matrix[t, k * 3 + 2] = lm.z / scale_factor

    return matrix


def _compute_body_scale(sequence) -> float:
    """计算身体尺度因子（平均躯干长度），用于归一化体型差异"""
    # 肩中点(11,12) → 髋中点(23,24)
    total_torso = 0.0
    count = 0
    for frame in sequence.frames:
        landmarks = frame.landmarks
        if len(landmarks) <= 24:
            continue
        # 肩中点
        sx = (landmarks[11].x + landmarks[12].x) / 2
        sy = (landmarks[11].y + landmarks[12].y) / 2
        # 髋中点
        hx = (landmarks[23].x + landmarks[24].x) / 2
        hy = (landmarks[23].y + landmarks[24].y) / 2
        torso = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        if torso > 0.01:
            total_torso += torso
            count += 1
    return total_torso / count if count > 0 else 1.0
