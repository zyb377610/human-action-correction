"""
MediaPipe 33 关键点骨骼拓扑图定义

为 ST-GCN 模型提供人体骨骼邻接矩阵和边列表。
基于 MediaPipe Pose Landmark 的 33 个关键点连接关系。
"""

from typing import List, Tuple

import numpy as np

from src.pose_estimation.data_types import POSE_CONNECTIONS, NUM_LANDMARKS


def get_edge_list() -> List[Tuple[int, int]]:
    """
    获取骨骼连接边列表

    Returns:
        边列表，每个元素为 (起点索引, 终点索引) 元组
    """
    return list(POSE_CONNECTIONS)


def get_adjacency_matrix(self_loop: bool = True) -> np.ndarray:
    """
    获取骨骼邻接矩阵

    Args:
        self_loop: 是否添加自连接（对角线为 1）

    Returns:
        形状为 (33, 33) 的对称邻接矩阵
    """
    adj = np.zeros((NUM_LANDMARKS, NUM_LANDMARKS), dtype=np.float32)

    # 填充边
    for i, j in POSE_CONNECTIONS:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # 自连接
    if self_loop:
        np.fill_diagonal(adj, 1.0)

    return adj


def get_normalized_adjacency() -> np.ndarray:
    """
    获取对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}

    用于 ST-GCN 中的图卷积运算，确保特征聚合时的数值稳定性。

    Returns:
        形状为 (33, 33) 的归一化邻接矩阵
    """
    adj = get_adjacency_matrix(self_loop=True)

    # 度矩阵 D
    degree = np.sum(adj, axis=1)  # (33,)

    # D^{-1/2}
    d_inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])

    # D^{-1/2} A D^{-1/2}
    d_mat = np.diag(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat

    return norm_adj.astype(np.float32)