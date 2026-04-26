"""
数据增强工具

支持时间拉伸/压缩、高斯噪声、左右镜像翻转和批量增强。
用于扩充标准动作模板数据集。
"""

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from src.pose_estimation.data_types import PoseLandmark, PoseFrame, PoseSequence

logger = logging.getLogger(__name__)

# MediaPipe 33 关键点的左右对称映射
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6),         # 眼睛
    (7, 8),                           # 耳
    (9, 10),                          # 嘴
    (11, 12),                         # 肩
    (13, 14),                         # 肘
    (15, 16),                         # 腕
    (17, 18), (19, 20), (21, 22),    # 手
    (23, 24),                         # 髋
    (25, 26),                         # 膝
    (27, 28),                         # 踝
    (29, 30), (31, 32),              # 脚
]


def _sequence_to_array(sequence: PoseSequence) -> np.ndarray:
    """PoseSequence -> (T, 33, 4) ndarray [x, y, z, visibility]"""
    T = sequence.num_frames
    arr = np.zeros((T, 33, 4), dtype=np.float64)
    for t, frame in enumerate(sequence.frames):
        for j, lm in enumerate(frame.landmarks):
            arr[t, j] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr


def _array_to_sequence(
    arr: np.ndarray, fps: float, original_sequence: Optional[PoseSequence] = None
) -> PoseSequence:
    """(T, 33, 4) ndarray -> PoseSequence"""
    T = arr.shape[0]
    frames = []
    for t in range(T):
        landmarks = [
            PoseLandmark(
                x=float(arr[t, j, 0]),
                y=float(arr[t, j, 1]),
                z=float(arr[t, j, 2]),
                visibility=float(arr[t, j, 3]),
            )
            for j in range(33)
        ]
        frames.append(PoseFrame(
            timestamp=t / fps,
            frame_index=t,
            landmarks=landmarks,
        ))
    return PoseSequence(frames=frames, fps=fps)


# ===== 3.1 时间拉伸/压缩 =====

def time_warp(
    sequence: PoseSequence,
    speed_factor: float = 1.0,
) -> PoseSequence:
    """
    时间拉伸/压缩

    Args:
        sequence: 输入序列
        speed_factor: 速度因子。>1 加速（帧数减少），<1 减速（帧数增加）

    Returns:
        变速后的 PoseSequence
    """
    if speed_factor <= 0:
        raise ValueError(f"speed_factor 必须 > 0，收到 {speed_factor}")

    arr = _sequence_to_array(sequence)  # (T, 33, 4)
    T = arr.shape[0]
    new_T = max(2, int(round(T / speed_factor)))

    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_T)

    new_arr = np.zeros((new_T, 33, 4), dtype=np.float64)
    for j in range(33):
        for c in range(4):
            f = interp1d(old_t, arr[:, j, c], kind="linear")
            new_arr[:, j, c] = f(new_t)

    # visibility 裁剪到 [0, 1]
    new_arr[:, :, 3] = np.clip(new_arr[:, :, 3], 0.0, 1.0)

    logger.debug(f"time_warp: {T} -> {new_T} 帧 (factor={speed_factor})")
    return _array_to_sequence(new_arr, sequence.fps)


# ===== 3.2 高斯噪声 =====

def add_noise(
    sequence: PoseSequence,
    noise_std: float = 0.005,
    seed: Optional[int] = None,
) -> PoseSequence:
    """
    对关键点坐标添加高斯噪声

    Args:
        sequence: 输入序列
        noise_std: 噪声标准差（归一化坐标系下）
        seed: 随机种子，用于可复现

    Returns:
        添加噪声后的 PoseSequence
    """
    rng = np.random.default_rng(seed)
    arr = _sequence_to_array(sequence)  # (T, 33, 4)

    # 仅对 x, y, z (前 3 个通道) 添加噪声，visibility 不变
    noise = rng.normal(0, noise_std, size=(arr.shape[0], 33, 3))
    arr[:, :, :3] += noise

    logger.debug(f"add_noise: std={noise_std}")
    return _array_to_sequence(arr, sequence.fps)


# ===== 3.3 左右镜像翻转 =====

def mirror_sequence(sequence: PoseSequence) -> PoseSequence:
    """
    左右镜像翻转

    交换左右对称关键点，并对 x 坐标取镜像 (1 - x)。

    Returns:
        镜像后的 PoseSequence
    """
    arr = _sequence_to_array(sequence)  # (T, 33, 4)

    # 1. x 坐标镜像 (MediaPipe 归一化坐标范围 [0, 1])
    arr[:, :, 0] = 1.0 - arr[:, :, 0]

    # 2. 交换左右关键点
    for left, right in LEFT_RIGHT_PAIRS:
        arr[:, left, :], arr[:, right, :] = (
            arr[:, right, :].copy(),
            arr[:, left, :].copy(),
        )

    logger.debug("mirror_sequence: 左右镜像完成")
    return _array_to_sequence(arr, sequence.fps)


# ===== 3.4 批量增强 =====

def augment_batch(
    sequence: PoseSequence,
    num_augmented: int = 5,
    time_warp_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.005,
    mirror_enabled: bool = True,
    seed: Optional[int] = None,
) -> List[PoseSequence]:
    """
    组合多种增强策略，批量生成增强样本

    策略组合:
      1. 原始序列 + 噪声
      2. 时间拉伸 + 噪声
      3. 镜像 + 噪声
      4. 镜像 + 时间拉伸 + 噪声

    Args:
        sequence: 原始标准序列
        num_augmented: 要生成的增强样本数
        time_warp_range: (min_speed, max_speed) 速度因子范围
        noise_std: 高斯噪声标准差
        mirror_enabled: 是否启用镜像增强
        seed: 随机种子

    Returns:
        增强后的 PoseSequence 列表
    """
    rng = np.random.default_rng(seed)
    results: List[PoseSequence] = []

    for i in range(num_augmented):
        aug = sequence

        # 随机时间拉伸
        speed = rng.uniform(time_warp_range[0], time_warp_range[1])
        if abs(speed - 1.0) > 0.02:  # 避免太接近 1.0 的无效变换
            aug = time_warp(aug, speed)

        # 随机决定是否镜像
        if mirror_enabled and rng.random() > 0.5:
            aug = mirror_sequence(aug)

        # 添加噪声
        aug = add_noise(aug, noise_std, seed=int(rng.integers(0, 2**31)))

        results.append(aug)

    logger.info(f"augment_batch: 生成 {len(results)} 个增强样本")
    return results
