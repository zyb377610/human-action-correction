"""
距离度量函数

支持欧氏距离、余弦距离、曼哈顿距离，提供统一接口。
"""

from typing import Callable, Optional, Sequence

import numpy as np


# ===== 核心关节点索引 =====
# 从 MediaPipe 33 个关键点中筛选 12 个运动核心关节点，
# 去掉头部（0-10）、手指（17-22）、脚趾（29-32）等噪声点。
# 这些核心关节直接决定动作质量，用于 DTW 比对时效果更好。
CORE_JOINT_INDICES = [
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

# 核心关节点中文名称
CORE_JOINT_NAMES = {
    11: "左肩", 12: "右肩",
    13: "左肘", 14: "右肘",
    15: "左腕", 16: "右腕",
    23: "左髋", 24: "右髋",
    25: "左膝", 26: "右膝",
    27: "左踝", 28: "右踝",
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
) -> np.ndarray:
    """
    将 PoseSequence 转换为特征矩阵

    默认只使用 12 个核心关节点的 (x, y, z) 坐标，
    去掉头部、手指、脚趾等噪声点，提升 DTW 比对精度。

    Args:
        sequence: PoseSequence 对象
        joint_indices: 要使用的关节点索引列表，
                       None 时使用 CORE_JOINT_INDICES（12 个核心关节）

    Returns:
        (T, num_joints * 3) ndarray
        默认: (T, 36)  即 12 个关节 × 3 维坐标
    """
    if joint_indices is None:
        joint_indices = CORE_JOINT_INDICES

    T = sequence.num_frames
    num_joints = len(joint_indices)
    matrix = np.zeros((T, num_joints * 3), dtype=np.float64)

    for t, frame in enumerate(sequence.frames):
        for k, j in enumerate(joint_indices):
            if j < len(frame.landmarks):
                lm = frame.landmarks[j]
                matrix[t, k * 3] = lm.x
                matrix[t, k * 3 + 1] = lm.y
                matrix[t, k * 3 + 2] = lm.z

    return matrix
