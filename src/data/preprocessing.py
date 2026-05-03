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
            interpolated = interp_func(missing_indices)
            # 防护：外推可能产生 NaN/Inf，用最近有效值替代
            bad = ~np.isfinite(interpolated)
            if bad.any():
                interpolated[bad] = np.mean(values)
            result[missing_indices, j, ch] = interpolated

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
    含有 NaN/Inf 的通道会先填充再滤波。

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
            channel = arr[:, j, ch].copy()
            # 防护：处理 NaN/Inf 值
            bad_mask = ~np.isfinite(channel)
            if bad_mask.any():
                if bad_mask.all():
                    # 全部无效，跳过滤波
                    continue
                # 用有效值的均值填充无效位置
                valid_mean = np.nanmean(channel[~bad_mask])
                channel[bad_mask] = valid_mean
            result[:, j, ch] = savgol_filter(
                channel, window_length, polyorder
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


def filter_skeleton_outliers(
    sequence: PoseSequence,
    scale_jump_threshold: float = 0.4,
    position_jump_threshold: float = 0.3,
) -> PoseSequence:
    """
    过滤骨骼突变帧（多人遮挡干扰检测）

    当检测到躯干尺度或身体中心位置的帧间突变超过阈值时，
    认为该帧检测到了错误的人物（如遮挡者），将其标记为无效。

    原理：
    - 同一个人的躯干长度（肩中点到髋中点）在相邻帧之间变化极小
    - 同一个人的身体中心位置不会突然跳变
    - 遮挡或换人会导致这两个指标突变

    Args:
        sequence: 输入姿态序列
        scale_jump_threshold: 躯干尺度变化率阈值（相对值），超过此值视为突变
        position_jump_threshold: 身体中心跳变阈值（归一化坐标距离）

    Returns:
        过滤后的新 PoseSequence（移除了异常帧）
    """
    T = sequence.num_frames
    if T < 3:
        return sequence

    # 计算每帧的躯干长度和身体中心
    torso_lengths = np.zeros(T, dtype=np.float64)
    body_centers = np.zeros((T, 2), dtype=np.float64)

    for t, frame in enumerate(sequence.frames):
        lm = frame.landmarks
        if len(lm) <= 24:
            continue
        # 肩中点
        sx = (lm[11].x + lm[12].x) / 2
        sy = (lm[11].y + lm[12].y) / 2
        # 髋中点
        hx = (lm[23].x + lm[24].x) / 2
        hy = (lm[23].y + lm[24].y) / 2
        torso_lengths[t] = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        body_centers[t] = [(sx + hx) / 2, (sy + hy) / 2]

    # 计算中位数躯干长度作为基准
    valid_torsos = torso_lengths[torso_lengths > 0.01]
    if len(valid_torsos) < 3:
        return sequence
    median_torso = np.median(valid_torsos)

    # 标记有效帧
    valid_mask = np.ones(T, dtype=bool)
    for t in range(T):
        # 检查躯干尺度是否偏离中位数太远
        if torso_lengths[t] > 0.01:
            scale_ratio = abs(torso_lengths[t] - median_torso) / median_torso
            if scale_ratio > scale_jump_threshold:
                valid_mask[t] = False
                continue

        # 检查与前一帧的身体中心跳变
        if t > 0 and valid_mask[t - 1] and torso_lengths[t] > 0.01:
            center_jump = np.sqrt(np.sum((body_centers[t] - body_centers[t - 1]) ** 2))
            if center_jump > position_jump_threshold:
                valid_mask[t] = False

    # 构建过滤后的序列
    removed_count = int(np.sum(~valid_mask))
    if removed_count == 0:
        return sequence

    filtered = PoseSequence(fps=sequence.fps, metadata=dict(sequence.metadata))
    filtered.metadata['skeleton_outliers_removed'] = removed_count
    new_idx = 0
    for t in range(T):
        if valid_mask[t]:
            # 重建连续帧索引和时间戳，避免后续插值出现NaN
            old_frame = sequence.frames[t]
            new_frame = PoseFrame(
                timestamp=new_idx / sequence.fps,
                frame_index=new_idx,
                landmarks=old_frame.landmarks,
            )
            filtered.add_frame(new_frame)
            new_idx += 1

    logger.info(
        f"骨骼突变过滤: {T}帧 → {filtered.num_frames}帧 "
        f"(移除 {removed_count} 帧异常骨骼)"
    )
    return filtered


def extract_action_segment(
    sequence: PoseSequence,
    min_frames: int = 10,
    smooth_window: int = 5,
    energy_percentile: float = 25,
    min_energy_threshold: float = 0.001,
) -> PoseSequence:
    """
    从长视频序列中自动提取动作发生的片段（增强版）

    通过计算逐帧运动能量（所有核心关节位移之和），
    找到连续高能量区域作为"动作片段"，剔除准备/收尾等无关帧。

    增强点（相比初版）：
    - 更智能的阈值策略：结合中位数和百分位数
    - 支持合并相邻的高能量段（允许短暂停顿）
    - 能量最高段优先（而非最长段）
    - 可调参数更灵活

    原理：
    - 用户录制视频通常包含：站定准备 → 执行动作 → 站定收尾
    - 非动作段的运动能量接近 0，动作段能量显著升高
    - 取能量最高的连续区域作为动作片段

    Args:
        sequence: 完整视频的姿态序列
        min_frames: 最小帧数（低于此值不裁剪）
        smooth_window: 能量曲线平滑窗口
        energy_percentile: 能量阈值百分位数
        min_energy_threshold: 最小能量阈值

    Returns:
        裁剪后的 PoseSequence（如果无法检测则返回原序列）
    """
    T = sequence.num_frames
    if T < min_frames * 2:
        return sequence

    from src.action_comparison.distance_metrics import CORE_JOINT_INDICES

    # 1. 计算逐帧运动能量（相邻帧核心关节的总位移）
    energy = np.zeros(T - 1, dtype=np.float64)
    prev_positions = None
    for t, frame in enumerate(sequence.frames):
        positions = []
        for j in CORE_JOINT_INDICES:
            if j < len(frame.landmarks):
                lm = frame.landmarks[j]
                if lm.visibility > 0.3:
                    positions.append((lm.x, lm.y))
        if not positions:
            continue
        positions = np.array(positions)
        if prev_positions is not None and len(positions) == len(prev_positions):
            energy[t - 1] = np.mean(np.sqrt(np.sum((positions - prev_positions) ** 2, axis=1)))
        prev_positions = positions

    if len(energy) == 0 or np.max(energy) < 1e-6:
        return sequence

    # 2. 平滑能量曲线
    if smooth_window > 1 and len(energy) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        energy = np.convolve(energy, kernel, mode='same')

    # 3. 智能阈值：使用中位数和百分位数的组合
    median_energy = np.median(energy)
    percentile_energy = np.percentile(energy, energy_percentile)
    # 取两者中较小的，但不低于最小阈值
    threshold = max(min(median_energy * 0.5, percentile_energy), min_energy_threshold)

    above = energy > threshold

    # 4. 找到所有连续的高能量段，允许短暂间隙（合并相邻段）
    gap_tolerance = max(3, int(T * 0.02))  # 允许的最大间隙帧数
    segments = []
    start = None
    gap_count = 0

    for i, val in enumerate(above):
        if val:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    end = i - gap_count
                    if end - start >= min_frames // 2:  # 放宽最小段长度
                        segments.append((start, end))
                    start = None
                    gap_count = 0

    if start is not None:
        end = len(above) if above[-1] else len(above) - gap_count
        if end - start >= min_frames // 2:
            segments.append((start, end))

    if not segments:
        return sequence  # 无有效片段，返回原序列

    # 5. 选择最佳段：优先选能量最高的段（而非最长的）
    def segment_score(seg):
        s, e = seg
        seg_energy = np.sum(energy[s:e])
        seg_length = e - s
        # 综合评分 = 总能量 × 长度权重（避免选太短的高能段）
        length_weight = min(1.0, seg_length / (T * 0.3))
        return seg_energy * (0.5 + 0.5 * length_weight)

    best = max(segments, key=segment_score)
    start_idx = max(0, best[0] - 5)   # 前留 5 帧余量
    end_idx = min(T, best[1] + 6)     # 后留 6 帧余量

    # 如果裁剪后长度不够或裁剪比例太小（几乎没裁剪），返回原序列
    if end_idx - start_idx < min_frames:
        return sequence

    # 如果裁剪后还是几乎全部帧，没必要裁剪
    if (end_idx - start_idx) >= T * 0.95:
        return sequence

    # 6. 构建裁剪后的序列
    cropped = PoseSequence(fps=sequence.fps, metadata=dict(sequence.metadata))
    cropped.metadata['cropped_from'] = (start_idx, end_idx)
    cropped.metadata['original_frames'] = T

    for i in range(start_idx, end_idx):
        if i < sequence.num_frames:
            cropped.add_frame(sequence.frames[i])

    logger.info(
        f"动作片段提取: {T}帧 → {cropped.num_frames}帧 "
        f"(帧 {start_idx}-{end_idx}, "
        f"能量阈值={threshold:.4f})"
    )
    return cropped


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
