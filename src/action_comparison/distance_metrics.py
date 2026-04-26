"""
距离度量函数

支持欧氏距离、余弦距离、曼哈顿距离，提供统一接口。
"""

from typing import Callable

import numpy as np


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


def sequence_to_feature_matrix(sequence) -> np.ndarray:
    """
    将 PoseSequence 转换为 (T, 33*3) 特征矩阵

    每帧 33 个关键点的 (x, y, z) 坐标拼接为 99 维向量。

    Args:
        sequence: PoseSequence 对象

    Returns:
        (T, 99) ndarray
    """
    T = sequence.num_frames
    matrix = np.zeros((T, 33 * 3), dtype=np.float64)
    for t, frame in enumerate(sequence.frames):
        for j, lm in enumerate(frame.landmarks):
            matrix[t, j * 3] = lm.x
            matrix[t, j * 3 + 1] = lm.y
            matrix[t, j * 3 + 2] = lm.z
    return matrix
