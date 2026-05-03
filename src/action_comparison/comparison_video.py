"""
矫正对比视频生成器

沿 DTW 对齐路径生成逐帧骨骼对比视频，左右并排：
- 左半：用户原始帧 + 偏差颜色骨骼叠加（🟢正常 🟡轻微 🔴需矫正）
- 右半：同步的标准模板骨架动画（深色底）
- 底部信息栏：帧号、评分、偏差值、矫正建议
"""

import logging
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
)
from src.action_comparison.distance_metrics import CORE_JOINT_INDICES, CORE_JOINT_NAMES
from .comparison import ComparisonResult

logger = logging.getLogger(__name__)

# ===== 偏差颜色映射（BGR） =====
# 阈值基于归一化坐标 [0,1] 中的欧氏距离
# 放宽阈值以使颜色分布更合理
_THRESHOLD_MILD = 0.08     # < 8% 图像尺寸 → 绿色
_THRESHOLD_MODERATE = 0.18  # 8%-18% → 黄色，> 18% → 红色
COLOR_GOOD = (46, 204, 113)       # 绿色
COLOR_MODERATE = (44, 156, 243)   # 黄色
COLOR_BAD = (52, 73, 231)         # 红色
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)      # 模板骨架灰色
COLOR_BG = (35, 35, 40)           # 深灰背景
COLOR_PANEL_BG = (28, 28, 32)     # 右侧面板背景
COLOR_DIVIDER = (80, 80, 80)      # 分隔线
COLOR_INFO_BG = (20, 22, 26)      # 底部信息栏背景

# 关节绘制参数
_RADIUS_RATIO = 0.009
_THICKNESS_RATIO = 0.0035

# 默认输出参数
_DEFAULT_FPS = 15
_DEFAULT_CODEC = "avc1"

# ---- 中文关节名 → MediaPipe 索引 ----
_CHINESE_NAME_TO_INDEX: Dict[str, int] = {}
for _idx, _cn_name in CORE_JOINT_NAMES.items():
    _CHINESE_NAME_TO_INDEX[_cn_name] = _idx
# 也支持英文名回退
for _idx, _en_name in enumerate(LANDMARK_NAMES):
    if _idx in CORE_JOINT_INDICES and _en_name not in _CHINESE_NAME_TO_INDEX:
        _CHINESE_NAME_TO_INDEX[_en_name] = _idx


def _get_joint_color(deviation: float) -> Tuple[int, int, int]:
    """根据偏差值返回 BGR 颜色"""
    if deviation < _THRESHOLD_MILD:
        return COLOR_GOOD
    elif deviation < _THRESHOLD_MODERATE:
        return COLOR_MODERATE
    else:
        return COLOR_BAD


def _build_deviation_map(
    joint_deviations: Dict[str, float]
) -> Dict[int, Tuple[float, Tuple[int, int, int]]]:
    """
    构建 关节索引 → (偏差, 颜色) 映射

    joint_deviations 的 key 可能是中文名（"左肩"）或英文名（"left_shoulder"），
    通过 _CHINESE_NAME_TO_INDEX 统一转为 MediaPipe 索引。
    """
    result: Dict[int, Tuple[float, Tuple[int, int, int]]] = {}
    for name, dev in joint_deviations.items():
        idx = _CHINESE_NAME_TO_INDEX.get(name)
        if idx is None:
            # 尝试英文名
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

    Args:
        video_path: 用户原始视频路径
        user_sequence: 用户 PoseSequence
        template_sequence: 模板 PoseSequence
        comparison_result: DTW 对比结果（含对齐路径 path）
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

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- 2. 构建 DTW 对齐映射 ----
    # 关键：DTW 路径的索引对应的是归一化后 60 帧的序列，不是原始帧。
    # 需要做 原始帧 → 归一化帧 → DTW路径 → 模板归一化帧 → 模板原始帧 的映射。
    path = comparison_result.path
    DTW_NORM_FRAMES = 60  # 与 preprocessing 默认 target_frames 一致

    user_orig_total = user_sequence.num_frames
    tpl_orig_total = template_sequence.num_frames

    # norm_user_idx → norm_tpl_idx
    norm_user_to_tpl: Dict[int, int] = {}
    for norm_u, norm_t in path:
        norm_user_to_tpl[norm_u] = norm_t

    # original_user_idx → original_tpl_idx
    def _orig_to_tpl(orig_user_idx: int) -> Optional[int]:
        """将原始用户帧索引映射到原始模板帧索引"""
        if user_orig_total <= 0 or tpl_orig_total <= 0:
            return None
        norm_u = min(DTW_NORM_FRAMES - 1, int(orig_user_idx * DTW_NORM_FRAMES / user_orig_total))
        norm_t = norm_user_to_tpl.get(norm_u)
        if norm_t is None:
            return None
        return min(tpl_orig_total - 1, int(norm_t * tpl_orig_total / DTW_NORM_FRAMES))

    # ---- 3. 准备模板骨架数据 ----
    tpl_frames_data: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for i, frame in enumerate(template_sequence.frames):
        tpl_data = []
        for lm in frame.landmarks:
            tpl_data.append((lm.x, lm.y, lm.z, lm.visibility))
        tpl_frames_data[i] = tpl_data

    # ---- 4. 构建偏差颜色映射 ----
    joint_deviation_map = _build_deviation_map(joint_deviations)
    logger.info(f"偏差关节映射: {len(joint_deviation_map)} 个关节")

    # ---- 5. 准备矫正建议 ----
    correction_lines: List[str] = []
    if corrections:
        for c in corrections[:8]:
            if hasattr(c, 'joint_display_name') and hasattr(c, 'advice'):
                correction_lines.append(f"{c.joint_display_name}: {c.advice}")
            elif hasattr(c, 'description'):
                correction_lines.append(str(c.description))

    # ---- 6. 确定输出布局 ----
    # 左侧：用户视频，右侧：模板骨架面板（等宽）
    panel_w = video_w  # 左右各占原始视频宽度
    info_bar_h = int(video_h * 0.16)
    output_w = video_w * 2  # 左右并排
    output_h = video_h + info_bar_h
    divider_x = video_w     # 分隔线 X 坐标

    # ---- 7. 写入视频 ----
    fourcc = cv2.VideoWriter_fourcc(*_DEFAULT_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
        if not writer.isOpened():
            cap.release()
            logger.error(f"无法创建输出视频: {output_path}")
            return None

    # ---- 8. 逐帧生成 ----
    frame_count = min(total_video_frames, user_sequence.num_frames)
    step = max(1, frame_count // 100)

    for user_idx in range(frame_count):
        ret, raw_frame = cap.read()
        if not ret:
            break

        # 创建画布
        canvas = np.full((output_h, output_w, 3), COLOR_BG, dtype=np.uint8)

        # -- 左半：用户视频 --
        canvas[:video_h, :video_w] = raw_frame

        # 绘制用户骨骼
        if user_idx < user_sequence.num_frames:
            user_pose = user_sequence.frames[user_idx]
            _draw_skeleton_on_region(
                canvas, user_pose, 0, 0, video_w, video_h,
                joint_deviation_map, is_template=False,
            )

        # 标签
        cv2.putText(canvas, "Your Action", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)

        # -- 右半：模板骨架（DTW 对齐）--
        # 右边填充面板背景
        canvas[:video_h, video_w:output_w] = COLOR_PANEL_BG

        tpl_idx = _orig_to_tpl(user_idx)
        if tpl_idx is not None and tpl_idx in tpl_frames_data:
            _draw_skeleton_on_region(
                canvas, None, video_w, 0, video_w, video_h,
                joint_deviation_map=None, is_template=True,
                tpl_landmarks=tpl_frames_data[tpl_idx],
            )

        # 标签
        cv2.putText(canvas, "Template (DTW matched)", (video_w + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 2)

        # 显示模板帧号
        if tpl_idx is not None:
            tpl_text = f"Tpl frame: {tpl_idx + 1}/{tpl_orig_total}"
            (tw, _), _ = cv2.getTextSize(tpl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(canvas, tpl_text,
                        (output_w - tw - 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

        # -- 中间分隔线 --
        cv2.line(canvas, (divider_x, 0), (divider_x, video_h), COLOR_DIVIDER, 2)

        # -- 底部信息栏 --
        _draw_info_bar(
            canvas, video_h, output_w, output_h,
            user_idx=user_idx, total_frames=frame_count,
            quality_score=quality_score,
            joint_deviations=joint_deviations,
            correction_lines=correction_lines,
            joint_deviation_map=joint_deviation_map,
        )

        writer.write(canvas)

        if progress_callback and user_idx % step == 0:
            progress_callback(user_idx + 1, frame_count)

    cap.release()
    writer.release()

    logger.info(f"对比视频已生成: {output_path} ({frame_count} 帧, {output_w}x{output_h})")
    return str(output_path)


def _draw_skeleton_on_region(
    canvas: np.ndarray,
    pose: Optional[PoseFrame],
    offset_x: int, offset_y: int,
    region_w: int, region_h: int,
    joint_deviation_map: Optional[Dict[int, Tuple[float, Tuple[int, int, int]]]],
    is_template: bool = False,
    tpl_landmarks: Optional[List[Tuple[float, float, float, float]]] = None,
):
    """
    在画布的指定区域绘制骨骼

    - 用户模式：按偏差着色，关键点 + 连线
    - 模板模式：灰色骨架
    """
    radius = max(2, int(region_h * _RADIUS_RATIO))
    thickness = max(1, int(region_h * _THICKNESS_RATIO))

    # 构建像素坐标
    px: Dict[int, Tuple[int, int]] = {}
    vis: Dict[int, float] = {}

    if is_template and tpl_landmarks is not None:
        # 模板模式：从传入的坐标列表绘制
        for idx in DISPLAY_JOINT_INDICES:
            if idx < len(tpl_landmarks):
                x, y, _, v = tpl_landmarks[idx]
                px[idx] = (offset_x + int(x * region_w), offset_y + int(y * region_h))
                vis[idx] = v
    elif pose is not None:
        # 用户模式：从 PoseFrame 绘制
        landmarks = pose.landmarks
        for idx in DISPLAY_JOINT_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px[idx] = (offset_x + int(lm.x * region_w),
                          offset_y + int(lm.y * region_h))
                vis[idx] = lm.visibility

    if not px:
        return

    # 获取颜色函数
    if is_template:
        joint_color = COLOR_GRAY
        conn_color = (100, 100, 100)
    else:
        def joint_color(idx: int):
            if joint_deviation_map and idx in joint_deviation_map:
                return joint_deviation_map[idx][1]
            return COLOR_GOOD

        def conn_color(idx_a: int, idx_b: int):
            if joint_deviation_map:
                da = joint_deviation_map.get(idx_a, (0, COLOR_GOOD))[0]
                db = joint_deviation_map.get(idx_b, (0, COLOR_GOOD))[0]
                return _get_joint_color((da + db) / 2)
            return COLOR_GOOD

    # 绘制连线
    for a, b in DISPLAY_CONNECTIONS:
        if a in px and b in px and vis.get(a, 0) > VISIBILITY_THRESHOLD and vis.get(b, 0) > VISIBILITY_THRESHOLD:
            color = conn_color if not is_template else conn_color
            if not is_template:
                color = conn_color(a, b)
            cv2.line(canvas, px[a], px[b], color, thickness)

    # 绘制关键点
    for idx in DISPLAY_JOINT_INDICES:
        if idx in px and vis.get(idx, 0) > VISIBILITY_THRESHOLD:
            color = joint_color if is_template else joint_color(idx)
            cv2.circle(canvas, px[idx], radius, color, -1)
            cv2.circle(canvas, px[idx], radius, COLOR_WHITE, 1)


def _draw_info_bar(
    canvas: np.ndarray,
    frame_top: int, canvas_w: int, canvas_h: int,
    user_idx: int, total_frames: int,
    quality_score: float,
    joint_deviations: Dict[str, float],
    correction_lines: List[str],
    joint_deviation_map: Dict[int, Tuple[float, Tuple[int, int, int]]],
):
    """绘制底部信息栏（纯 ASCII，避免 OpenCV 中文乱码）"""
    bar_top = frame_top
    bar_h = canvas_h - frame_top

    # 背景
    cv2.rectangle(canvas, (0, bar_top), (canvas_w, canvas_h), COLOR_INFO_BG, -1)
    cv2.line(canvas, (0, bar_top), (canvas_w, bar_top), COLOR_DIVIDER, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x_margin = 10
    line_h = max(14, bar_h // 4)

    # ---- 第1行：帧号 + 评分 ----
    y = bar_top + line_h + 2
    cv2.putText(canvas, f"Frame: {user_idx + 1}/{total_frames}",
                (x_margin, y), font, 0.4, COLOR_WHITE, 1)

    score_text = f"Score: {quality_score:.1f}/100"
    score_color = COLOR_GOOD if quality_score >= 70 else (COLOR_MODERATE if quality_score >= 40 else COLOR_BAD)
    cv2.putText(canvas, score_text,
                (canvas_w // 2 - 50, y), font, 0.45, score_color, 1)

    # 图例
    legend_x = canvas_w - 260
    cv2.putText(canvas, "G:OK", (legend_x, y), font, 0.35, COLOR_GOOD, 1)
    cv2.putText(canvas, "Y:Mild", (legend_x + 60, y), font, 0.35, COLOR_MODERATE, 1)
    cv2.putText(canvas, "R:Bad", (legend_x + 130, y), font, 0.35, COLOR_BAD, 1)

    # ---- 第2行：偏差最大的关节（英文缩写）----
    y += line_h
    if joint_deviations:
        sorted_items = sorted(joint_deviations.items(), key=lambda x: x[1], reverse=True)[:8]
        parts = []
        for name, dev in sorted_items:
            short = _short_joint_name(name)
            tag = "G" if dev < _THRESHOLD_MILD else ("Y" if dev < _THRESHOLD_MODERATE else "R")
            parts.append(f"[{tag}]{short}:{dev:.3f}")
        text = " | ".join(parts)
        max_chars = 130
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        cv2.putText(canvas, text, (x_margin, y), font, 0.32, (190, 190, 190), 1)

    # ---- 第3行：矫正建议轮播（过滤中文） ----
    y += line_h
    if correction_lines:
        line_idx = (user_idx // 40) % len(correction_lines)
        text = _sanitize_ascii(correction_lines[line_idx])
        max_chars = 140
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        cv2.putText(canvas, f"> {text}",
                    (x_margin, y), font, 0.33, (180, 210, 255), 1)


def _short_joint_name(name: str) -> str:
    """将关节名转为简短 ASCII 缩写"""
    mapping = {
        "nose": "nose",
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
        "鼻子": "nose",
    }
    return mapping.get(name, name[:6])


def _sanitize_ascii(text: str) -> str:
    """过滤非 ASCII 字符，保留英文/数字/标点"""
    return ''.join(c if ord(c) < 128 else '?' for c in text)


# ===== 保留从 numpy 数组生成的接口 =====

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
    """从 numpy 数组生成对比视频（用于摄像头录制模式）"""
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

