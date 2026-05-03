"""
DTW 算法实现

支持经典 DTW、FastDTW、Derivative DTW 三种变体。
所有算法通过 compute_dtw() 统一调度。
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from fastdtw import fastdtw as _fastdtw

from .distance_metrics import euclidean_distance, get_distance_func

logger = logging.getLogger(__name__)


def classic_dtw(
    query: np.ndarray,
    template: np.ndarray,
    dist_func: Callable = euclidean_distance,
    window_size: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]], np.ndarray]:
    """
    经典 DTW 算法

    时间复杂度 O(n*m)，支持 Sakoe-Chiba 带宽约束。

    Args:
        query: (N, D) 查询序列
        template: (M, D) 模板序列
        dist_func: 距离函数 (a, b) -> float
        window_size: Sakoe-Chiba 带宽，None 表示不限制

    Returns:
        (distance, path, cost_matrix)
        - distance: DTW 距离
        - path: 最优对齐路径 [(i, j), ...]
        - cost_matrix: (N, M) 累积代价矩阵
    """
    N, M = len(query), len(template)

    # 累积代价矩阵
    cost = np.full((N, M), np.inf, dtype=np.float64)

    # 确定搜索窗口
    if window_size is not None:
        w = max(window_size, abs(N - M))  # 保证能到达 (N-1, M-1)
    else:
        w = max(N, M)  # 无约束

    # 填充代价矩阵
    cost[0, 0] = dist_func(query[0], template[0])

    for i in range(N):
        j_start = max(0, i - w)
        j_end = min(M, i + w + 1)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            d = dist_func(query[i], template[j])
            candidates = []
            if i > 0 and j > 0:
                candidates.append(cost[i - 1, j - 1])
            if i > 0:
                candidates.append(cost[i - 1, j])
            if j > 0:
                candidates.append(cost[i, j - 1])
            cost[i, j] = d + min(candidates) if candidates else d

    distance = float(cost[N - 1, M - 1])

    # 回溯最优路径
    path = _backtrack(cost, N, M)

    return distance, path, cost


def _backtrack(cost: np.ndarray, N: int, M: int) -> List[Tuple[int, int]]:
    """从累积代价矩阵回溯最优对齐路径"""
    path = [(N - 1, M - 1)]
    i, j = N - 1, M - 1

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (cost[i - 1, j - 1], i - 1, j - 1),
                (cost[i - 1, j], i - 1, j),
                (cost[i, j - 1], i, j - 1),
            ]
            _, ni, nj = min(candidates, key=lambda x: x[0])
            i, j = ni, nj
        path.append((i, j))

    path.reverse()
    return path


def subsequence_dtw(
    query: np.ndarray,
    template: np.ndarray,
    dist_func: Callable = euclidean_distance,
    window_size: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]], np.ndarray, int, int]:
    """
    子序列 DTW（Open-End DTW）

    当 query（用户视频）比 template（标准模板）长时，
    在 query 中寻找与 template 最佳匹配的连续子序列。

    原理：
    - 模板可以匹配到 query 的任意连续段落
    - cost[i, 0] = d(q_i, t_0)（不累积，允许从任意位置开始）
    - 最优终点 = argmin_i cost[i, M-1]
    - 回溯从该终点开始，自然截取匹配段

    Args:
        query: (N, D) 长序列（用户）
        template: (M, D) 短序列（模板）

    Returns:
        (distance, path, cost_matrix, start_idx, end_idx)
    """
    N, M = len(query), len(template)

    if N < M:
        # query 更短，回退经典 DTW
        dist, path, cost = classic_dtw(query, template, dist_func, window_size)
        return dist, path, cost, 0, N - 1

    cost = np.full((N, M), np.inf, dtype=np.float64)

    # 第一列：可从任意 query 帧开始，不累积
    for i in range(N):
        cost[i, 0] = dist_func(query[i], template[0])

    # 第一行：正常累积
    for j in range(1, M):
        cost[0, j] = cost[0, j - 1] + dist_func(query[0], template[j])

    # DP 主体
    for i in range(1, N):
        for j in range(1, M):
            d = dist_func(query[i], template[j])
            cost[i, j] = d + min(cost[i - 1, j - 1],
                                 cost[i - 1, j],
                                 cost[i, j - 1])

    # 找到最佳终点（query 中模板结束的位置）
    best_end = int(np.argmin(cost[:, M - 1]))
    distance = float(cost[best_end, M - 1])

    # 回溯最优路径（从 best_end 开始）
    path = _subsequence_backtrack(cost, best_end, N, M)

    # 提取匹配段的起止
    start_idx = path[0][0] if path else 0
    end_idx = best_end

    return distance, path, cost, start_idx, end_idx


def _subsequence_backtrack(
    cost: np.ndarray, end_i: int, N: int, M: int
) -> List[Tuple[int, int]]:
    """从指定终点回溯子序列 DTW 路径"""
    path = [(end_i, M - 1)]
    i, j = end_i, M - 1

    while j > 0:
        if i == 0:
            j -= 1
        else:
            candidates = []
            if j > 0:
                candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
            candidates.append((cost[i - 1, j], i - 1, j))
            if j > 0:
                candidates.append((cost[i, j - 1], i, j - 1))
            _, ni, nj = min(candidates, key=lambda x: x[0])
            i, j = ni, nj
        path.append((i, j))

    path.reverse()
    return path


def fast_dtw(
    query: np.ndarray,
    template: np.ndarray,
    dist_func: Callable = euclidean_distance,
    radius: int = 1,
) -> Tuple[float, List[Tuple[int, int]], Optional[np.ndarray]]:
    """
    FastDTW 近似算法

    基于 fastdtw 库，线性时间复杂度 O(n)。

    Args:
        query: (N, D) 查询序列
        template: (M, D) 模板序列
        dist_func: 距离函数
        radius: FastDTW 近似半径

    Returns:
        (distance, path, None)
        注意: FastDTW 不返回完整代价矩阵
    """
    distance, path = _fastdtw(query, template, radius=radius, dist=dist_func)
    # fastdtw 返回的 path 是 list of tuples
    path = [tuple(p) for p in path]
    return float(distance), path, None


def derivative_dtw(
    query: np.ndarray,
    template: np.ndarray,
    dist_func: Callable = euclidean_distance,
    window_size: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]], np.ndarray]:
    """
    Derivative DTW (DDTW)

    对序列求一阶差分后再执行经典 DTW。
    更侧重形状匹配而非绝对位置。

    边界处理: 首帧复制填充保持长度不变。

    Args:
        query: (N, D) 查询序列
        template: (M, D) 模板序列
        dist_func: 距离函数
        window_size: Sakoe-Chiba 带宽约束

    Returns:
        (distance, path, cost_matrix)
    """
    query_d = _compute_derivative(query)
    template_d = _compute_derivative(template)
    return classic_dtw(query_d, template_d, dist_func, window_size)


def _compute_derivative(seq: np.ndarray) -> np.ndarray:
    """
    计算一阶差分（导数估计）

    使用中间差分: d[i] = ((q[i]-q[i-1]) + (q[i+1]-q[i-1])/2) / 2
    边界: d[0] = d[1], d[-1] = d[-2]
    """
    N = len(seq)
    if N < 3:
        return np.zeros_like(seq)

    d = np.zeros_like(seq)
    for i in range(1, N - 1):
        d[i] = ((seq[i] - seq[i - 1]) + (seq[i + 1] - seq[i - 1]) / 2) / 2

    # 边界填充
    d[0] = d[1]
    d[-1] = d[-2]
    return d


# ===== 统一调度 =====

def compute_dtw(
    query: np.ndarray,
    template: np.ndarray,
    algorithm: str = "dtw",
    metric: str = "euclidean",
    window_size: Optional[int] = None,
    use_subsequence: bool = False,
) -> Tuple[float, List[Tuple[int, int]], Optional[np.ndarray]]:
    """
    统一 DTW 调度函数

    Args:
        query: (N, D) 查询序列特征矩阵
        template: (M, D) 模板序列特征矩阵
        algorithm: "dtw" / "fastdtw" / "ddtw"
        metric: "euclidean" / "cosine" / "manhattan"
        window_size: Sakoe-Chiba 带宽约束
        use_subsequence: 是否使用子序列 DTW（query 长于 template 时启用）

    Returns:
        (distance, path, cost_matrix)
    """
    dist_func = get_distance_func(metric)
    algorithm = algorithm.lower().strip()

    if use_subsequence and len(query) > len(template) * 1.1:
        # 子序列模式：模板在 query 中找最佳匹配段
        distance, path, cost, _, _ = subsequence_dtw(
            query, template, dist_func, window_size
        )
        return distance, path, cost

    if algorithm == "dtw":
        return classic_dtw(query, template, dist_func, window_size)
    elif algorithm == "fastdtw":
        radius = window_size if window_size else 1
        return fast_dtw(query, template, dist_func, radius)
    elif algorithm == "ddtw":
        return derivative_dtw(query, template, dist_func, window_size)
    else:
        raise ValueError(
            f"不支持的 DTW 算法: '{algorithm}'，可选: dtw / fastdtw / ddtw"
        )
