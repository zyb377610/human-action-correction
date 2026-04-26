"""
数据预处理模块

对姿态关键点序列进行预处理：
- 缺失值插值（低可见度关键点）
- Savitzky-Golay 平滑滤波
- 序列长度归一化（重采样到固定帧数）
- 预处理流水线（按顺序执行上述步骤）
"""

import logging
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from src.pose_estimation.data_types import (
    PoseLandmark, PoseFrame, PoseSequence, NUM_LANDMARKS
)

logger = logging.getLogger(__name__)

# 默认可见度阈值
DEFAULT_VISIBILITY_THRESHOLD = 0.5
# 默认目标帧数
DEFAULT_TARGET_FRAMES = 60
# Savitzky-Golay 默认参数
DEFAULT_WINDOW_LENGTH = 7
DEFAULT_POLYORDER = 3


def interpolate_missing(
    sequence: PoseSequence,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> PoseSequence:
    """
    对低可见度的关键点进行线性插值填充

    遍历每个关键点通道，将 visibility < 阈值的帧标记为缺失，
    用前后有效帧进行线性插值填充坐标值，并将 visibility 设为阈值。

    Args:
        sequence: 输入姿态序列
        visibility_threshold: 可见度阈值，低于此值视为缺失

    Returns:
        插值后的新 PoseSequence
    """
    if sequence.num_frames < 2:
        return sequence

    arr = sequence.to_numpy()  # (T, 33, 4) → [x, y, z, visibility]
    T = arr.shape[0]
    result = arr.copy()

    for j in range(NUM_LANDMARKS):
        vis = arr[:, j, 3]  # visibility 通道
        valid_mask = vis >= visibility_threshold

        if valid_mask.all() or not valid_mask.any():
            continue  # 全部有效或全部缺失，跳过

        valid_indices = np.where(valid_mask)[0]
        missing_indices = np.where(~valid_mask)[0]

        # 对 x, y, z 三个通道分别插值
        for ch in range(3):
            values = arr[valid_indices, j, ch]
            interp_func = interp1d(
                valid_indices, values,
                kind='linear', fill_value='extrapolate'
            )
            result[missing_indices, j, ch] = interp_func(missing_indices)

        # 将缺失帧的 visibility 设为阈值
        result[missing_indices, j, 3] = visibility_threshold

    return _array_to_sequence(result, sequence)


def smooth_sequence(
    sequence: PoseSequence,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    polyorder: int = DEFAULT_POLYORDER,
) -> PoseSequence:
    """
    使用 Savitzky-Golay 滤波器对关键点序列平滑去噪

    对每个关键点的 x, y, z 通道分别进行滤波。

    Args:
        sequence: 输入姿态序列
        window_length: 滤波窗口长度（必须为奇数）
        polyorder: 多项式阶数

    Returns:
        平滑后的新 PoseSequence
    """
    if sequence.num_frames < window_length:
        logger.warning(
            f"序列帧数({sequence.num_frames}) < 窗口长度({window_length})，跳过滤波"
        )
        return sequence

    arr = sequence.to_numpy()  # (T, 33, 4)
    result = arr.copy()

    for j in range(NUM_LANDMARKS):
        for ch in range(3):  # x, y, z
            result[:, j, ch] = savgol_filter(
                arr[:, j, ch], window_length, polyorder
            )

    return _array_to_sequence(result, sequence)


def resample_sequence(
    sequence: PoseSequence,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> PoseSequence:
    """
    将序列重采样到固定帧数

    使用线性插值将不等长序列归一化到统一长度，便于后续对比和训练。

    Args:
        sequence: 输入姿态序列
        target_frames: 目标帧数

    Returns:
        重采样后的新 PoseSequence
    """
    if sequence.num_frames < 2:
        return sequence

    arr = sequence.to_numpy()  # (T, 33, 4)
    T = arr.shape[0]

    src_indices = np.linspace(0, 1, T)
    dst_indices = np.linspace(0, 1, target_frames)

    result = np.zeros((target_frames, NUM_LANDMARKS, 4), dtype=np.float32)

    for j in range(NUM_LANDMARKS):
        for ch in range(4):  # x, y, z, visibility
            interp_func = interp1d(src_indices, arr[:, j, ch], kind='linear')
            result[:, j, ch] = interp_func(dst_indices)

    # 构建新序列
    new_fps = target_frames / sequence.duration if sequence.duration > 0 else sequence.fps
    new_seq = PoseSequence(fps=new_fps, metadata=dict(sequence.metadata))
    new_seq.metadata['original_frames'] = T
    new_seq.metadata['resampled_to'] = target_frames

    for i in range(target_frames):
        timestamp = i / new_fps if new_fps > 0 else 0.0
        landmarks = [
            PoseLandmark(
                x=float(result[i, j, 0]),
                y=float(result[i, j, 1]),
                z=float(result[i, j, 2]),
                visibility=float(result[i, j, 3]),
            )
            for j in range(NUM_LANDMARKS)
        ]
        new_seq.add_frame(PoseFrame(
            timestamp=timestamp,
            frame_index=i,
            landmarks=landmarks,
        ))

    return new_seq


def preprocess_pipeline(
    sequence: PoseSequence,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    smooth_window: int = DEFAULT_WINDOW_LENGTH,
    smooth_polyorder: int = DEFAULT_POLYORDER,
    target_frames: Optional[int] = DEFAULT_TARGET_FRAMES,
) -> PoseSequence:
    """
    数据预处理流水线：插值 → 滤波 → 归一化

    Args:
        sequence: 输入姿态序列
        visibility_threshold: 插值的可见度阈值
        smooth_window: 滤波窗口长度
        smooth_polyorder: 滤波多项式阶数
        target_frames: 目标帧数，None 时跳过重采样

    Returns:
        预处理后的新 PoseSequence
    """
    logger.info(f"预处理开始: {sequence.num_frames} 帧")

    # Step 1: 缺失值插值
    result = interpolate_missing(sequence, visibility_threshold)
    logger.info(f"  插值完成: {result.num_frames} 帧")

    # Step 2: 平滑滤波
    result = smooth_sequence(result, smooth_window, smooth_polyorder)
    logger.info(f"  滤波完成: {result.num_frames} 帧")

    # Step 3: 序列长度归一化
    if target_frames is not None:
        result = resample_sequence(result, target_frames)
        logger.info(f"  重采样完成: {result.num_frames} 帧")

    logger.info(f"预处理完成: {result.num_frames} 帧")
    return result


def _array_to_sequence(arr: np.ndarray, ref_seq: PoseSequence) -> PoseSequence:
    """将 NumPy 数组转回 PoseSequence，保留原序列的时间戳和元信息"""
    new_seq = PoseSequence(fps=ref_seq.fps, metadata=dict(ref_seq.metadata))
    T = arr.shape[0]

    for i in range(T):
        if i < len(ref_seq.frames):
            ts = ref_seq.frames[i].timestamp
            idx = ref_seq.frames[i].frame_index
        else:
            ts = i / ref_seq.fps
            idx = i

        landmarks = [
            PoseLandmark(
                x=float(arr[i, j, 0]),
                y=float(arr[i, j, 1]),
                z=float(arr[i, j, 2]),
                visibility=float(arr[i, j, 3]),
            )
            for j in range(NUM_LANDMARKS)
        ]
        new_seq.add_frame(PoseFrame(timestamp=ts, frame_index=idx, landmarks=landmarks))

    return new_seq
