"""
矫正对比视频生成器

左右并排骨骼对比视频：
- 左半：用户原始视频帧 + 骨骼叠加（按偏差着色）
- 右半：模板骨架动画（深色底），DTW 对齐同步播放

核心逻辑：
- 用 PoseFrame.frame_index 精确匹配视频帧和骨骼数据（解决丢帧错位问题）
- 右侧模板通过 DTW 对齐路径驱动：用户做到哪个动作，模板就展示对应帧
- 动作段之外：模板暂停（首帧/末帧），让用户看清准备和收尾
- 帧率不同、时长不同的情况：DTW 自动处理非线性时间扭曲
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pose_estimation.data_types import (
    PoseSequence, PoseFrame, PoseLandmark,
    LANDMARK_NAMES,
)
from src.pose_estimation.visualizer import (
    DISPLAY_JOINT_INDICES, DISPLAY_CONNECTIONS,
    VISIBILITY_THRESHOLD,
    put_chinese_text,
)
from src.pose_estimation.feature_extractor import get_joint_angles, JOINT_ANGLE_DEFINITIONS
from src.action_comparison.distance_metrics import CORE_JOINT_INDICES, CORE_JOINT_NAMES
from .comparison import ComparisonResult

logger = logging.getLogger(__name__)

# ===== 颜色常量（BGR） =====
_THRESHOLD_MILD = 0.08
_THRESHOLD_MODERATE = 0.18
COLOR_GOOD = (46, 204, 113)
COLOR_MODERATE = (44, 156, 243)
COLOR_BAD = (52, 73, 231)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_BG = (35, 35, 40)
COLOR_PANEL_BG = (28, 28, 32)
COLOR_DIVIDER = (80, 80, 80)
COLOR_INFO_BG = (20, 22, 26)

_RADIUS_RATIO = 0.009
_THICKNESS_RATIO = 0.0035
_DEFAULT_FPS = 15
_DEFAULT_CODEC = "avc1"

# 中文关节名 → MediaPipe 索引
_CHINESE_NAME_TO_INDEX: Dict[str, int] = {}
for _idx, _cn_name in CORE_JOINT_NAMES.items():
    _CHINESE_NAME_TO_INDEX[_cn_name] = _idx
for _idx, _en_name in enumerate(LANDMARK_NAMES):
    if _idx in CORE_JOINT_INDICES and _en_name not in _CHINESE_NAME_TO_INDEX:
        _CHINESE_NAME_TO_INDEX[_en_name] = _idx


def _get_joint_color(deviation: float) -> Tuple[int, int, int]:
    if deviation < _THRESHOLD_MILD:
        return COLOR_GOOD
    elif deviation < _THRESHOLD_MODERATE:
        return COLOR_MODERATE
    else:
        return COLOR_BAD


def _build_direct_dtw_map(
    user_sequence: PoseSequence,
    template_sequence: PoseSequence,
) -> Tuple[Dict[int, int], int, int]:
    """
    在完整序列上运行子序列 DTW，构建帧映射。

    不对任何序列做 extract_action_segment 裁剪——让子序列 DTW
    自己找到模板在用户序列中的最佳匹配段。

    Args:
        user_sequence: 原始用户 PoseSequence（含 video frame_index）
        template_sequence: 原始模板 PoseSequence（完整使用）

    Returns:
        (frame_map, match_start, match_end)
        - frame_map: {user_seq_local_idx: template_seq_local_idx}
        - match_start: DTW 匹配段在 user_sequence 中的起始本地索引
        - match_end:   DTW 匹配段在 user_sequence 中的结束本地索引
    """
    from src.action_comparison.distance_metrics import sequence_to_feature_matrix
    from src.action_comparison.dtw_algorithms import compute_dtw

    user_n = user_sequence.num_frames
    tpl_n = template_sequence.num_frames

    if user_n < 3 or tpl_n < 3:
        logger.warning("帧数不足，回退线性映射")
        return {}, 0, user_n - 1

    # 1. 构建特征矩阵（身体比例归一化）——两个序列都不裁剪
    user_feat = sequence_to_feature_matrix(user_sequence, normalize_body_scale=True)
    tpl_feat = sequence_to_feature_matrix(template_sequence, normalize_body_scale=True)

    # 2. 子序列 DTW：在完整用户序列中寻找模板的最佳匹配段
    try:
        _, path, _ = compute_dtw(
            user_feat, tpl_feat,
            algorithm="dtw",
            metric="euclidean",
            use_subsequence=True,
        )
    except Exception as e:
        logger.warning(f"DTW 计算失败: {e}，回退线性映射")
        return {}, 0, user_n - 1

    if not path:
        return {}, 0, user_n - 1

    # 3. 提取 DTW 匹配段在用户序列中的范围
    match_start = min(p[0] for p in path)
    match_end = max(p[0] for p in path)

    # 4. 构建帧映射：user_seq 本地索引 → 完整模板索引
    q_to_t_list: Dict[int, List[int]] = defaultdict(list)
    for q, t in path:
        q_to_t_list[q].append(t)

    frame_map: Dict[int, int] = {}
    for q, t_list in q_to_t_list.items():
        frame_map[q] = int(np.median(t_list))

    logger.info(
        f"DTW 帧映射: {len(frame_map)} 对, "
        f"用户={user_n}帧, 模板={tpl_n}帧（均完整使用）, "
        f"匹配段=[{match_start}, {match_end}], 路径长度={len(path)}"
    )

    return frame_map, match_start, match_end


def _build_deviation_map(
    joint_deviations: Dict[str, float]
) -> Dict[int, Tuple[float, Tuple[int, int, int]]]:
    result: Dict[int, Tuple[float, Tuple[int, int, int]]] = {}
    for name, dev in joint_deviations.items():
        idx = _CHINESE_NAME_TO_INDEX.get(name)
        if idx is None:
            try:
                idx = LANDMARK_NAMES.index(name)
            except ValueError:
                continue
        if idx in DISPLAY_JOINT_INDICES:
            result[idx] = (dev, _get_joint_color(dev))
    return result


def generate_comparison_video(
    video_path: str,
    user_sequence: PoseSequence,
    template_sequence: PoseSequence,
    comparison_result: ComparisonResult,
    joint_deviations: Dict[str, float],
    output_path: str,
    fps: float = _DEFAULT_FPS,
    quality_score: float = 0.0,
    corrections: Optional[List] = None,
    progress_callback=None,
) -> Optional[str]:
    """
    生成左右并排的骨骼对比视频

    左侧：用户原始视频 + 骨骼叠加（精确按 frame_index 匹配）
    右侧：模板骨架动画（DTW 对齐驱动）
      - 用户准备阶段（未开始运动）→ 模板暂停在首帧
      - 用户动作阶段 → 模板由 DTW 对齐路径驱动，与用户动作节奏同步
      - 用户结束阶段 → 模板暂停在末帧
      - 丢帧时 → 模板保持在上一个有效位置

    帧率不同 / 时长不同的处理：
      DTW 自动处理非线性时间扭曲，用户动作快时模板也快，
      动作慢时模板也慢，确保两边同一时刻展示对应的动作姿态。

    Args:
        video_path: 用户原始视频路径
        user_sequence: 用户 PoseSequence（姿态估计原始输出）
        template_sequence: 模板 PoseSequence
        comparison_result: DTW 对比结果
        joint_deviations: 关节偏差 {name: deviation}
        output_path: 输出 MP4 路径
        fps: 输出帧率
        quality_score: 质量评分 [0, 100]
        corrections: 矫正建议列表
        progress_callback: fn(current, total)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. 打开原始视频 ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return None

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- 2. 构建 frame_index → PoseFrame 查找表 ----
    user_pose_lookup: Dict[int, PoseFrame] = {}
    for frame in user_sequence.frames:
        user_pose_lookup[frame.frame_index] = frame

    # ---- 3. 在完整序列上跑子序列 DTW，让 DTW 自己找匹配段 ----
    #     不预先裁剪任何序列，避免 extract_action_segment 过激裁剪
    #     导致 user_cropped < tpl_full 时子序列 DTW 退化为经典 DTW
    dtw_frame_map, match_start, match_end = _build_direct_dtw_map(
        user_sequence, template_sequence
    )

    # ---- 4. 准备模板骨架数据（完整使用） ----
    tpl_landmarks_list: List[List[Tuple[float, float, float, float]]] = []
    for frame in template_sequence.frames:
        lm_data = []
        for lm in frame.landmarks:
            lm_data.append((lm.x, lm.y, lm.z, lm.visibility))
        tpl_landmarks_list.append(lm_data)

    tpl_total = len(tpl_landmarks_list)

    # ---- 5. 偏差颜色映射 ----
    joint_deviation_map = _build_deviation_map(joint_deviations)
    logger.info(f"偏差关节映射: {len(joint_deviation_map)} 个关节")

    # ---- 6. 矫正建议 ----
    correction_lines: List[str] = []
    if corrections:
        for c in corrections[:8]:
            if hasattr(c, 'joint_display_name') and hasattr(c, 'advice'):
                correction_lines.append(f"{c.joint_display_name}: {c.advice}")
            elif hasattr(c, 'description'):
                correction_lines.append(str(c.description))

    # ---- 7. 输出布局 ----
    info_bar_h = int(video_h * 0.16)
    output_w = video_w * 2
    output_h = video_h + info_bar_h

    # ---- 8. 创建视频写入器 ----
    fourcc = cv2.VideoWriter_fourcc(*_DEFAULT_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
        if not writer.isOpened():
            cap.release()
            logger.error(f"无法创建输出视频: {output_path}")
            return None

    # ---- 9. 逐帧生成 ----
    frame_count = total_video_frames
    step = max(1, frame_count // 100)
    written = 0
    last_tpl_idx = 0

    # 将 DTW 匹配段的 user_seq 本地索引映射到 video frame_index
    # match_start / match_end 是 user_sequence 中的本地序号
    match_first_video_frame = (
        user_sequence.frames[match_start].frame_index
        if 0 <= match_start < user_sequence.num_frames else 0
    )
    match_last_video_frame = (
        user_sequence.frames[min(match_end, user_sequence.num_frames - 1)].frame_index
        if match_end >= match_start else frame_count
    )

    # 建立 user_seq 本地索引 → video frame_index 的快速查找
    seq_idx_to_video: Dict[int, int] = {}
    for seq_i, frame in enumerate(user_sequence.frames):
        seq_idx_to_video[seq_i] = frame.frame_index

    logger.info(
        f"DTW 匹配段: user_seq[{match_start}..{match_end}], "
        f"video帧 [{match_first_video_frame}..{match_last_video_frame}]"
    )

    for video_frame_idx in range(frame_count):
        ret, raw_frame = cap.read()
        if not ret:
            break

        canvas = np.full((output_h, output_w, 3), COLOR_BG, dtype=np.uint8)

        # ============ 左半：用户视频 + 骨骼 ============
        canvas[:video_h, :video_w] = raw_frame

        user_pose = user_pose_lookup.get(video_frame_idx, None)
        if user_pose is not None:
            _draw_skeleton_on_region(
                canvas, user_pose, 0, 0, video_w, video_h,
                joint_deviation_map, is_template=False,
            )

        cv2.putText(canvas, "Your Action", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)

        # ============ 右半：模板骨架（DTW 子序列匹配驱动） ============
        canvas[:video_h, video_w:output_w] = COLOR_PANEL_BG

        # 找到当前视频帧在 user_sequence 中的本地索引
        user_seq_idx = None
        for seq_i, frame in enumerate(user_sequence.frames):
            if frame.frame_index == video_frame_idx:
                user_seq_idx = seq_i
                break

        # ---- 三阶段：匹配前暂停 / 匹配中 DTW 同步 / 匹配后暂停 ----
        if user_seq_idx is not None and dtw_frame_map:
            if user_seq_idx < match_start:
                # 匹配前 → 模板停在首帧
                tpl_idx = 0
                phase_label = "Ready"
                phase_color = (120, 120, 120)
            elif user_seq_idx > match_end:
                # 匹配后 → 模板停在末帧
                tpl_idx = tpl_total - 1
                phase_label = "Done"
                phase_color = (100, 180, 100)
            else:
                # 匹配段内 → DTW 对齐（frame_map 键是 user_seq 本地索引）
                if user_seq_idx in dtw_frame_map:
                    tpl_idx = dtw_frame_map[user_seq_idx]
                else:
                    # 最近邻
                    all_keys = sorted(dtw_frame_map.keys())
                    nearest = min(all_keys, key=lambda k: abs(k - user_seq_idx))
                    tpl_idx = dtw_frame_map[nearest]
                tpl_idx = max(0, min(tpl_idx, tpl_total - 1))
                phase_label = "Synced"
                phase_color = (255, 180, 60)
        elif user_seq_idx is not None:
            # 无 DTW 映射 → 线性回退
            if video_frame_idx < match_first_video_frame:
                tpl_idx = 0
                phase_label = "Ready"
                phase_color = (120, 120, 120)
            elif video_frame_idx > match_last_video_frame:
                tpl_idx = tpl_total - 1
                phase_label = "Done"
                phase_color = (100, 180, 100)
            else:
                progress = (video_frame_idx - match_first_video_frame) / max(match_last_video_frame - match_first_video_frame, 1)
                tpl_idx = int(progress * (tpl_total - 1))
                tpl_idx = max(0, min(tpl_idx, tpl_total - 1))
                phase_label = "Linear"
                phase_color = (180, 140, 60)
        else:
            tpl_idx = last_tpl_idx
            phase_label = "Hold"
            phase_color = (100, 100, 100)

        last_tpl_idx = tpl_idx

        # 绘制模板骨骼
        _draw_skeleton_on_region(
            canvas, None, video_w, 0, video_w, video_h,
            joint_deviation_map=None, is_template=True,
            tpl_landmarks=tpl_landmarks_list[tpl_idx],
        )

        cv2.putText(canvas, f"Template [{phase_label}]", (video_w + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, phase_color, 2)

        # 模板帧号
        tpl_text = f"{tpl_idx + 1}/{tpl_total}"
        (tw, _), _ = cv2.getTextSize(tpl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, tpl_text,
                    (output_w - tw - 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

        # ============ 中间分隔线 ============
        cv2.line(canvas, (video_w, 0), (video_w, video_h), COLOR_DIVIDER, 2)

        # ============ 底部信息栏 ============
        _draw_info_bar(
            canvas, video_h, output_w, output_h,
            user_idx=video_frame_idx, total_frames=frame_count,
            quality_score=quality_score,
            joint_deviations=joint_deviations,
            correction_lines=correction_lines,
            joint_deviation_map=joint_deviation_map,
        )

        writer.write(canvas)
        written += 1

        if progress_callback and video_frame_idx % step == 0:
            progress_callback(video_frame_idx + 1, frame_count)

    cap.release()
    writer.release()

    logger.info(
        f"对比视频已生成: {output_path} ({written} 帧, {output_w}x{output_h}), "
        f"DTW匹配段: user_seq[{match_start}..{match_end}], 模板{ tpl_total}帧完整"
    )
    return str(output_path)


def _detect_action_range(
    user_sequence: PoseSequence,
    total_video_frames: int,
) -> Tuple[int, int]:
    """
    检测用户视频中动作发生的帧区间（在原始视频帧空间上）

    通过计算逐帧运动能量定位动作的起止位置。

    Returns:
        (action_start_frame, action_end_frame) 原始视频帧索引
    """
    T = user_sequence.num_frames
    if T < 5:
        return (0, total_video_frames - 1)

    # 收集每帧的 frame_index 和运动能量
    frame_indices = []
    for frame in user_sequence.frames:
        frame_indices.append(frame.frame_index)

    # 计算相邻帧的运动能量
    arr = user_sequence.to_numpy()  # (T, 33, 4)
    core_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    energy = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        diff = arr[t, core_joints, :3] - arr[t - 1, core_joints, :3]
        energy[t] = np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))

    # 平滑
    if T >= 5:
        kernel = np.ones(5) / 5
        energy = np.convolve(energy, kernel, mode='same')

    # 阈值：中位数的 0.5 倍，最低 0.001
    threshold = max(np.median(energy) * 0.5, 0.001)

    # 找到第一个超过阈值的帧和最后一个超过阈值的帧
    above = energy > threshold
    first_active = None
    last_active = None
    for i in range(T):
        if above[i]:
            if first_active is None:
                first_active = i
            last_active = i

    if first_active is None:
        # 没有检测到明显运动，使用全部帧
        return (0, total_video_frames - 1)

    # 前后留一点余量（3帧）
    first_active = max(0, first_active - 3)
    last_active = min(T - 1, last_active + 3)

    # 映射回原始视频帧索引
    action_start = frame_indices[first_active]
    action_end = frame_indices[last_active]

    # 保证合理范围
    action_start = max(0, action_start)
    action_end = min(total_video_frames - 1, action_end)

    return (action_start, action_end)


def _draw_skeleton_on_region(
    canvas: np.ndarray,
    pose: Optional[PoseFrame],
    offset_x: int, offset_y: int,
    region_w: int, region_h: int,
    joint_deviation_map: Optional[Dict[int, Tuple[float, Tuple[int, int, int]]]],
    is_template: bool = False,
    tpl_landmarks: Optional[List[Tuple[float, float, float, float]]] = None,
    show_labels: bool = True,
    show_angles: bool = True,
):
    """在画布指定区域绘制骨骼（含关节名称标签和角度值）"""
    radius = max(2, int(region_h * _RADIUS_RATIO))
    thickness = max(1, int(region_h * _THICKNESS_RATIO))

    px: Dict[int, Tuple[int, int]] = {}
    vis: Dict[int, float] = {}

    if is_template and tpl_landmarks is not None:
        for idx in DISPLAY_JOINT_INDICES:
            if idx < len(tpl_landmarks):
                x, y, _, v = tpl_landmarks[idx]
                px[idx] = (offset_x + int(x * region_w), offset_y + int(y * region_h))
                vis[idx] = v
    elif pose is not None:
        landmarks = pose.landmarks
        for idx in DISPLAY_JOINT_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px[idx] = (offset_x + int(lm.x * region_w),
                          offset_y + int(lm.y * region_h))
                vis[idx] = lm.visibility

    if not px:
        return

    # 颜色
    if is_template:
        joint_c = COLOR_GRAY
        conn_c = (100, 100, 100)
    else:
        joint_c = None
        conn_c = None

    # 连线
    for a, b in DISPLAY_CONNECTIONS:
        if a in px and b in px and vis.get(a, 0) > VISIBILITY_THRESHOLD and vis.get(b, 0) > VISIBILITY_THRESHOLD:
            if is_template:
                color = conn_c
            else:
                if joint_deviation_map:
                    da = joint_deviation_map.get(a, (0, COLOR_GOOD))[0]
                    db = joint_deviation_map.get(b, (0, COLOR_GOOD))[0]
                    color = _get_joint_color((da + db) / 2)
                else:
                    color = COLOR_GOOD
            cv2.line(canvas, px[a], px[b], color, thickness)

    # 关键点
    for idx in DISPLAY_JOINT_INDICES:
        if idx in px and vis.get(idx, 0) > VISIBILITY_THRESHOLD:
            if is_template:
                color = joint_c
            else:
                if joint_deviation_map and idx in joint_deviation_map:
                    color = joint_deviation_map[idx][1]
                else:
                    color = COLOR_GOOD
            cv2.circle(canvas, px[idx], radius, color, -1)
            cv2.circle(canvas, px[idx], radius, COLOR_WHITE, 1)

    # 关节名称标签（PIL 渲染中文）
    if show_labels:
        _draw_joint_labels_on_canvas(canvas, px, vis, region_w)

    # 关节角度标注
    if show_angles:
        _draw_angles_on_canvas(canvas, px, vis, region_w, pose, tpl_landmarks)


def _draw_joint_labels_on_canvas(
    canvas: np.ndarray,
    px: Dict[int, Tuple[int, int]],
    vis: Dict[int, float],
    region_w: int,
):
    """在已绘制的骨骼上添加中文关节名称标签（PIL 渲染）"""
    font_size = max(10, min(14, int(region_w / 45)))
    text_color = (220, 220, 220)

    for idx, name in CORE_JOINT_NAMES.items():
        if idx in px and vis.get(idx, 0) > VISIBILITY_THRESHOLD:
            x, y = px[idx]
            label_x = x + 5
            label_y = y + 3

            put_chinese_text(
                canvas, name,
                position=(label_x, label_y),
                font_size=font_size,
                text_color=text_color,
                bg_color=(0, 0, 0, 140),
            )


def _draw_angles_on_canvas(
    canvas: np.ndarray,
    px: Dict[int, Tuple[int, int]],
    vis: Dict[int, float],
    region_w: int,
    pose: Optional[PoseFrame],
    tpl_landmarks: Optional[List[Tuple[float, float, float, float]]],
):
    """在骨骼顶点旁标注关节角度值（黄色数字，°）"""
    font_size = max(9, min(12, int(region_w / 55)))

    # 构建角度计算所需的 landmarks
    angles = {}
    if pose is not None:
        angles = get_joint_angles(pose)
    elif tpl_landmarks is not None:
        # 从 tpl_landmarks 构造临时 PoseFrame 来计算角度
        from src.pose_estimation.data_types import PoseLandmark, PoseFrame as PF
        lms = [
            PoseLandmark(
                x=tpl_landmarks[j][0] if j < len(tpl_landmarks) else 0,
                y=tpl_landmarks[j][1] if j < len(tpl_landmarks) else 0,
                z=tpl_landmarks[j][2] if j < len(tpl_landmarks) else 0,
                visibility=tpl_landmarks[j][3] if j < len(tpl_landmarks) else 0,
            )
            for j in range(33)
        ]
        tmp_frame = PF(timestamp=0, frame_index=0, landmarks=lms)
        angles = get_joint_angles(tmp_frame)

    if not angles:
        return

    for name, angle_value in angles.items():
        if angle_value is None or (isinstance(angle_value, float) and (angle_value != angle_value)):
            continue
        # 获取顶点索引
        _, mid_idx, _ = JOINT_ANGLE_DEFINITIONS[name]
        if mid_idx not in px or vis.get(mid_idx, 0) < VISIBILITY_THRESHOLD:
            continue

        x, y = px[mid_idx]
        text_x = x + 5
        text_y = y - 10

        angle_text = f"{angle_value:.0f}"
        put_chinese_text(
            canvas, angle_text,
            position=(text_x, text_y),
            font_size=font_size,
            text_color=(0, 255, 255),
            bg_color=(0, 0, 0, 140),
        )


def _draw_info_bar(
    canvas: np.ndarray,
    frame_top: int, canvas_w: int, canvas_h: int,
    user_idx: int, total_frames: int,
    quality_score: float,
    joint_deviations: Dict[str, float],
    correction_lines: List[str],
    joint_deviation_map: Dict[int, Tuple[float, Tuple[int, int, int]]],
):
    """绘制底部信息栏"""
    bar_top = frame_top

    cv2.rectangle(canvas, (0, bar_top), (canvas_w, canvas_h), COLOR_INFO_BG, -1)
    cv2.line(canvas, (0, bar_top), (canvas_w, bar_top), COLOR_DIVIDER, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x_margin = 10
    bar_h = canvas_h - frame_top
    line_h = max(14, bar_h // 4)

    # 第1行：帧号 + 评分
    y = bar_top + line_h + 2
    cv2.putText(canvas, f"Frame: {user_idx + 1}/{total_frames}",
                (x_margin, y), font, 0.4, COLOR_WHITE, 1)

    score_text = f"Score: {quality_score:.1f}/100"
    score_color = COLOR_GOOD if quality_score >= 70 else (COLOR_MODERATE if quality_score >= 40 else COLOR_BAD)
    cv2.putText(canvas, score_text,
                (canvas_w // 2 - 50, y), font, 0.45, score_color, 1)

    legend_x = canvas_w - 260
    cv2.putText(canvas, "G:OK", (legend_x, y), font, 0.35, COLOR_GOOD, 1)
    cv2.putText(canvas, "Y:Mild", (legend_x + 60, y), font, 0.35, COLOR_MODERATE, 1)
    cv2.putText(canvas, "R:Bad", (legend_x + 130, y), font, 0.35, COLOR_BAD, 1)

    # 第2行：偏差关节
    y += line_h
    if joint_deviations:
        sorted_items = sorted(joint_deviations.items(), key=lambda x: x[1], reverse=True)[:8]
        parts = []
        for name, dev in sorted_items:
            short = _short_joint_name(name)
            tag = "G" if dev < _THRESHOLD_MILD else ("Y" if dev < _THRESHOLD_MODERATE else "R")
            parts.append(f"[{tag}]{short}:{dev:.3f}")
        text = " | ".join(parts)
        if len(text) > 130:
            text = text[:127] + "..."
        cv2.putText(canvas, text, (x_margin, y), font, 0.32, (190, 190, 190), 1)

    # 第3行：矫正建议
    y += line_h
    if correction_lines:
        line_idx = (user_idx // 40) % len(correction_lines)
        text = _sanitize_ascii(correction_lines[line_idx])
        if len(text) > 140:
            text = text[:137] + "..."
        cv2.putText(canvas, f"> {text}",
                    (x_margin, y), font, 0.33, (180, 210, 255), 1)


def _short_joint_name(name: str) -> str:
    mapping = {
        "nose": "nose", "鼻子": "nose",
        "左肩": "L.Shd", "left_shoulder": "L.Shd",
        "右肩": "R.Shd", "right_shoulder": "R.Shd",
        "左肘": "L.Elb", "left_elbow": "L.Elb",
        "右肘": "R.Elb", "right_elbow": "R.Elb",
        "左腕": "L.Wrst", "left_wrist": "L.Wrst",
        "右腕": "R.Wrst", "right_wrist": "R.Wrst",
        "左髋": "L.Hip", "left_hip": "L.Hip",
        "右髋": "R.Hip", "right_hip": "R.Hip",
        "左膝": "L.Knee", "left_knee": "L.Knee",
        "右膝": "R.Knee", "right_knee": "R.Knee",
        "左踝": "L.Ankl", "left_ankle": "L.Ankl",
        "右踝": "R.Ankl", "right_ankle": "R.Ankl",
        "左脚跟": "L.Heel", "left_heel": "L.Heel",
        "右脚跟": "R.Heel", "right_heel": "R.Heel",
        "左脚尖": "L.Toe", "left_foot_index": "L.Toe",
        "右脚尖": "R.Toe", "right_foot_index": "R.Toe",
    }
    return mapping.get(name, name[:6])


def _sanitize_ascii(text: str) -> str:
    return ''.join(c if ord(c) < 128 else '?' for c in text)


# ===== 兼容接口：从 numpy 数组生成 =====

def generate_comparison_video_from_arrays(
    user_frames: np.ndarray,
    template_frames: np.ndarray,
    original_video_path: str,
    output_path: str,
    comparison_result: ComparisonResult,
    joint_deviations: Dict[str, float],
    quality_score: float = 0.0,
    corrections: Optional[List] = None,
    fps: float = _DEFAULT_FPS,
    progress_callback=None,
) -> Optional[str]:
    """从 numpy 数组生成对比视频"""
    user_seq = PoseSequence(fps=fps)
    for i in range(user_frames.shape[0]):
        landmarks = []
        for j in range(user_frames.shape[1]):
            landmarks.append(PoseLandmark(
                x=float(user_frames[i, j, 0]),
                y=float(user_frames[i, j, 1]),
                z=float(user_frames[i, j, 2]),
                visibility=float(user_frames[i, j, 3]) if user_frames.shape[2] > 3 else 1.0,
            ))
        user_seq.add_frame(PoseFrame(timestamp=i / fps, frame_index=i, landmarks=landmarks))

    tpl_seq = PoseSequence(fps=fps)
    for i in range(template_frames.shape[0]):
        landmarks = []
        for j in range(template_frames.shape[1]):
            landmarks.append(PoseLandmark(
                x=float(template_frames[i, j, 0]),
                y=float(template_frames[i, j, 1]),
                z=float(template_frames[i, j, 2]),
                visibility=float(template_frames[i, j, 3]) if template_frames.shape[2] > 3 else 1.0,
            ))
        tpl_seq.add_frame(PoseFrame(timestamp=i / fps, frame_index=i, landmarks=landmarks))

    return generate_comparison_video(
        video_path=original_video_path,
        user_sequence=user_seq,
        template_sequence=tpl_seq,
        comparison_result=comparison_result,
        joint_deviations=joint_deviations,
        output_path=output_path,
        fps=fps,
        quality_score=quality_score,
        corrections=corrections,
        progress_callback=progress_callback,
    )
