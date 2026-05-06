"""
模板骨骼演示视频 + 帧对比渲染器

关键设计：
- 渲染骨骼的逻辑（线段颜色、关节点风格）完全复用
  `action_comparison.comparison_video._draw_skeleton_on_region`，
  保证视觉风格跟视频分析里看到的"骨骼对比视频"一致。
- 但在渲染前，会对**每一帧**做"居中 + 等比缩放到画布"的平移/缩放，
  避免原始拍摄画幅偏离中心导致骨骼挤在角落或显得过小。
  这个变换**只是平移 + 等比缩放**，不会改变骨骼形状（不会扭曲）。
- 不做任何旋转/归一化，骨骼形态与相机里看到的保持一致。
"""

import copy
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.pose_estimation.data_types import PoseFrame, PoseLandmark, PoseSequence
from src.pose_estimation.visualizer import (
    DISPLAY_JOINT_INDICES,
    VISIBILITY_THRESHOLD,
)
from src.action_comparison.comparison_video import (
    _draw_skeleton_on_region,
    COLOR_BG,
    COLOR_PANEL_BG,
    COLOR_WHITE,
)

logger = logging.getLogger(__name__)

# 画布尺寸
_DEFAULT_W = 480
_DEFAULT_H = 640
_DEFAULT_FPS = 20

# 编码器尝试顺序（跟 comparison_video 一致）
_CODEC_CANDIDATES = ("avc1", "H264", "mp4v")

# 画布边距（留 8% 边距，骨骼不顶边）
_PADDING = 0.08


# ============================================================================
#  居中 + 等比缩放
# ============================================================================

def _compute_sequence_fit_box(
    sequence: PoseSequence,
    padding: float = _PADDING,
) -> Tuple[float, float, float, float, float]:
    """
    计算整段序列的 landmarks 可见点的归一化边界框。

    返回 (cx, cy, half_w, half_h, scale_ref)，其中
    cx/cy = 整段序列可见点的中心
    half_w / half_h = 最大跨度的一半（保证整段动作都能装进去）
    scale_ref = 最大尺度，避免每帧缩放不一致造成骨骼"抖"

    这样做的效果是：
    - 骨骼永远居中
    - 整段动作中无论手伸得多开，骨骼都不会超出画布
    - 等比缩放 → 保持关节间相对关系不变，不会扭曲
    """
    xs: List[float] = []
    ys: List[float] = []
    for frame in sequence.frames:
        for lm in frame.landmarks:
            if lm.visibility >= VISIBILITY_THRESHOLD:
                xs.append(lm.x)
                ys.append(lm.y)

    if not xs or not ys:
        return 0.5, 0.5, 0.5, 0.5, 1.0

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half_w = max((xmax - xmin) / 2.0, 0.05)
    half_h = max((ymax - ymin) / 2.0, 0.05)
    # 稍微放宽一点做 padding
    half_w *= (1.0 + padding)
    half_h *= (1.0 + padding)
    scale_ref = max(half_w, half_h)
    return cx, cy, half_w, half_h, scale_ref


def _fit_frame_to_canvas(
    frame: PoseFrame,
    fit_box: Tuple[float, float, float, float, float],
) -> PoseFrame:
    """
    把一帧 landmarks 的 (x, y) 平移+等比缩放，使得该序列整体居中占满 [0,1]x[0,1]。
    z 保持原样（不参与渲染）。

    ⚠️ 只做线性平移 + 等比缩放 → 绝不扭曲骨骼形态。
    """
    cx, cy, _hw, _hh, scale_ref = fit_box
    # 目标：原始 (x, y) → (0.5 + (x - cx)/(2*scale_ref), 0.5 + (y - cy)/(2*scale_ref))
    # 这样保持宽高比不变（统一用 scale_ref 作分母），骨骼不变形。
    denom = 2.0 * scale_ref if scale_ref > 1e-6 else 1.0

    new_landmarks: List[PoseLandmark] = []
    for lm in frame.landmarks:
        nx = 0.5 + (lm.x - cx) / denom
        ny = 0.5 + (lm.y - cy) / denom
        # 夹紧到 [0, 1]，避免极端姿态溢出
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        new_landmarks.append(PoseLandmark(
            x=nx, y=ny, z=lm.z, visibility=lm.visibility,
        ))
    return PoseFrame(
        timestamp=frame.timestamp,
        frame_index=frame.frame_index,
        landmarks=new_landmarks,
    )


def _landmarks_of_frame(frame: PoseFrame):
    """抽取一帧 (x, y, z, visibility) 列表"""
    return [(lm.x, lm.y, lm.z, lm.visibility) for lm in frame.landmarks]


# ============================================================================
#  Video writer
# ============================================================================

def _open_writer(path: Path, fps: int, w: int, h: int):
    """依次尝试编码器"""
    for codec in _CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            logger.info(f"模板演示视频使用编码器 {codec}: {path.name}")
            return writer
        writer.release()
    return None


# ============================================================================
#  模板演示视频
# ============================================================================

def render_template_demo_video(
    sequence: PoseSequence,
    output_path: str,
    title: Optional[str] = None,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    fps: int = _DEFAULT_FPS,
) -> Optional[str]:
    """将 PoseSequence 渲染为一段骨骼动画 MP4（居中等比缩放）"""
    if sequence is None or sequence.num_frames == 0:
        logger.warning("render_template_demo_video: 空序列")
        return None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = _open_writer(out_path, fps, width, height)
    if writer is None:
        logger.error(f"VideoWriter 打开失败: {out_path}")
        return None

    # 整段序列统一的 fit_box，保证渲染期间比例稳定不抖动
    fit_box = _compute_sequence_fit_box(sequence)
    total = sequence.num_frames

    try:
        for i, frame in enumerate(sequence.frames):
            canvas = np.full((height, width, 3), COLOR_PANEL_BG, dtype=np.uint8)
            fitted = _fit_frame_to_canvas(frame, fit_box)
            tpl_lm = _landmarks_of_frame(fitted)
            _draw_skeleton_on_region(
                canvas, None, 0, 0, width, height,
                joint_deviation_map=None,
                is_template=True,
                tpl_landmarks=tpl_lm,
                show_labels=True,
                show_angles=True,
            )
            if title:
                cv2.putText(canvas, title, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{i + 1}/{total}",
                        (width - 80, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (160, 160, 160), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        logger.info(f"模板演示视频已生成: {out_path} ({total}帧@{fps}fps)")
        return str(out_path)
    except Exception as e:
        logger.error(f"模板演示视频生成失败: {e}", exc_info=True)
        try:
            writer.release()
        except Exception:
            pass
        return None


# ============================================================================
#  帧对比渲染
# ============================================================================

def render_pair_frame(
    user_frame: Optional[PoseFrame],
    template_frame: Optional[PoseFrame],
    user_fit_box: Optional[Tuple[float, float, float, float, float]],
    template_fit_box: Optional[Tuple[float, float, float, float, float]],
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    title_left: str = "Your Action",
    title_right: str = "Template",
    highlight_joints: Optional[list] = None,
) -> np.ndarray:
    """
    并排渲染一帧"用户 vs 模板"骨骼对比图。

    Args:
        user_frame / template_frame: 原始 PoseFrame
        user_fit_box / template_fit_box: 该序列的整段 fit_box（居中缩放参数）
        width / height: 单侧画幅
        title_left / title_right: 顶部标签
        highlight_joints: 用户侧需要红圈强调的关节索引列表

    Returns:
        BGR ndarray，尺寸 (height, width * 2, 3)
    """
    canvas = np.full((height, width * 2, 3), COLOR_BG, dtype=np.uint8)
    canvas[:, :width] = COLOR_PANEL_BG
    canvas[:, width:] = COLOR_PANEL_BG

    # 左：用户骨骼（绿色）
    fitted_user = None
    if user_frame is not None and user_fit_box is not None:
        fitted_user = _fit_frame_to_canvas(user_frame, user_fit_box)
        _draw_skeleton_on_region(
            canvas, fitted_user, 0, 0, width, height,
            joint_deviation_map=None,
            is_template=False,
            show_labels=True,
            show_angles=True,
        )

    # 右：模板骨骼（灰色）
    if template_frame is not None and template_fit_box is not None:
        fitted_tpl = _fit_frame_to_canvas(template_frame, template_fit_box)
        tpl_lm = _landmarks_of_frame(fitted_tpl)
        _draw_skeleton_on_region(
            canvas, None, width, 0, width, height,
            joint_deviation_map=None,
            is_template=True,
            tpl_landmarks=tpl_lm,
            show_labels=True,
            show_angles=True,
        )

    # 在左侧画红圈高亮偏差大的关节（使用缩放后的坐标）
    if highlight_joints and fitted_user is not None:
        for j in highlight_joints:
            if j < len(fitted_user.landmarks):
                lm = fitted_user.landmarks[j]
                if lm.visibility < VISIBILITY_THRESHOLD:
                    continue
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                cv2.circle(canvas, (cx, cy), 14, (60, 60, 255), 2, cv2.LINE_AA)

    # 顶部标签
    cv2.putText(canvas, title_left, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, title_right, (width + 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)

    # 分隔线
    cv2.line(canvas, (width, 0), (width, height), (70, 70, 80), 2)

    return canvas


# 暴露给 pipeline 用于预计算
compute_sequence_fit_box = _compute_sequence_fit_box
