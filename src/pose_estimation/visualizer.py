"""
姿态可视化模块

在视频帧上绘制骨骼连接图、关键点标注和关节角度信息。
只绘制 17 个核心关节点及其连线，去掉面部细节和手指等噪声点。
"""

import time
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

from .data_types import (
    PoseFrame, POSE_CONNECTIONS, LANDMARK_NAMES, NUM_LANDMARKS
)
from .feature_extractor import get_joint_angles, JOINT_ANGLE_DEFINITIONS


# ===== 要显示的 17 个核心关节点 =====
DISPLAY_JOINT_INDICES: Set[int] = {
    0,               # nose — 头部参考
    11, 12,           # 左右肩
    13, 14,           # 左右肘
    15, 16,           # 左右腕
    23, 24,           # 左右髋
    25, 26,           # 左右膝
    27, 28,           # 左右踝
    29, 30,           # 左右脚跟
    31, 32,           # 左右脚尖
}

# 只绘制核心关节之间的骨骼连线
DISPLAY_CONNECTIONS = [
    # 躯干
    (0, 11), (0, 12),                      # 鼻子 → 双肩（头部朝向）
    (11, 12),                              # 双肩
    (11, 23), (12, 24),                    # 肩 → 髋
    (23, 24),                              # 双髋
    # 左臂
    (11, 13), (13, 15),                    # 左肩 → 左肘 → 左腕
    # 右臂
    (12, 14), (14, 16),                    # 右肩 → 右肘 → 右腕
    # 左腿
    (23, 25), (25, 27),                    # 左髋 → 左膝 → 左踝
    (27, 29), (27, 31), (29, 31),          # 左脚
    # 右腿
    (24, 26), (26, 28),                    # 右髋 → 右膝 → 右踝
    (28, 30), (28, 32), (30, 32),          # 右脚
]


# ===== 默认绘制样式 =====
DEFAULT_LANDMARK_COLOR = (0, 255, 0)        # 绿色关键点
DEFAULT_LANDMARK_RADIUS = 4
LOW_VIS_LANDMARK_COLOR = (128, 128, 128)    # 灰色（低可见度）
LOW_VIS_LANDMARK_RADIUS = 3

DEFAULT_CONNECTION_COLOR = (0, 200, 0)      # 绿色连线
DEFAULT_CONNECTION_THICKNESS = 2
LOW_VIS_CONNECTION_COLOR = (80, 80, 80)     # 灰色（低可见度连线）

ANGLE_TEXT_COLOR = (0, 255, 255)            # 黄色角度文本
ANGLE_FONT_SCALE = 0.4
ANGLE_FONT_THICKNESS = 1

FPS_TEXT_COLOR = (0, 0, 255)                # 红色FPS
FPS_FONT_SCALE = 0.7
FPS_FONT_THICKNESS = 2

# 可见度阈值
VISIBILITY_THRESHOLD = 0.5


def draw_skeleton(
    image: np.ndarray,
    frame: PoseFrame,
    landmark_color: Tuple[int, int, int] = DEFAULT_LANDMARK_COLOR,
    landmark_radius: int = DEFAULT_LANDMARK_RADIUS,
    connection_color: Tuple[int, int, int] = DEFAULT_CONNECTION_COLOR,
    connection_thickness: int = DEFAULT_CONNECTION_THICKNESS,
    draw_low_visibility: bool = True,
) -> np.ndarray:
    """
    在图像上绘制 17 个核心关节的骨骼连接图

    只绘制核心关节点和它们之间的连线，
    去掉面部细节（眼睛/耳朵/嘴角）和手指等噪声点。

    Args:
        image: BGR 格式图像（会被修改）
        frame: 单帧姿态数据
        landmark_color: 关键点颜色 (B, G, R)
        landmark_radius: 关键点半径
        connection_color: 连线颜色 (B, G, R)
        connection_thickness: 连线粗细
        draw_low_visibility: 是否绘制低可见度的关键点

    Returns:
        绘制后的图像
    """
    h, w = image.shape[:2]
    landmarks = frame.landmarks

    # 1. 绘制核心骨骼连线
    for (i1, i2) in DISPLAY_CONNECTIONS:
        if i1 >= len(landmarks) or i2 >= len(landmarks):
            continue

        lm1, lm2 = landmarks[i1], landmarks[i2]

        both_visible = (lm1.visibility >= VISIBILITY_THRESHOLD and
                        lm2.visibility >= VISIBILITY_THRESHOLD)

        if not both_visible and not draw_low_visibility:
            continue

        pt1 = (int(lm1.x * w), int(lm1.y * h))
        pt2 = (int(lm2.x * w), int(lm2.y * h))

        color = connection_color if both_visible else LOW_VIS_CONNECTION_COLOR
        thickness = connection_thickness if both_visible else 1

        cv2.line(image, pt1, pt2, color, thickness)

    # 2. 只绘制核心关键点
    for i in DISPLAY_JOINT_INDICES:
        if i >= len(landmarks):
            continue

        lm = landmarks[i]
        is_visible = lm.visibility >= VISIBILITY_THRESHOLD

        if not is_visible and not draw_low_visibility:
            continue

        pt = (int(lm.x * w), int(lm.y * h))

        if is_visible:
            cv2.circle(image, pt, landmark_radius, landmark_color, -1)
        else:
            cv2.circle(image, pt, LOW_VIS_LANDMARK_RADIUS,
                       LOW_VIS_LANDMARK_COLOR, -1)

    return image


def draw_angles(
    image: np.ndarray,
    frame: PoseFrame,
    angles: Optional[Dict[str, float]] = None,
    text_color: Tuple[int, int, int] = ANGLE_TEXT_COLOR,
    font_scale: float = ANGLE_FONT_SCALE,
    font_thickness: int = ANGLE_FONT_THICKNESS,
) -> np.ndarray:
    """
    在关节位置标注角度值

    Args:
        image: BGR 格式图像
        frame: 单帧姿态数据
        angles: 关节角度字典，None 时自动计算
        text_color: 文本颜色
        font_scale: 字体大小
        font_thickness: 字体粗细

    Returns:
        标注后的图像
    """
    h, w = image.shape[:2]

    if angles is None:
        angles = get_joint_angles(frame)

    landmarks = frame.landmarks

    for name, angle_value in angles.items():
        if np.isnan(angle_value):
            continue

        # 获取关节顶点位置（中间点）
        _, mid_idx, _ = JOINT_ANGLE_DEFINITIONS[name]
        lm = landmarks[mid_idx]

        if lm.visibility < VISIBILITY_THRESHOLD:
            continue

        # 文本位置偏移，避免与关键点重叠
        text_x = int(lm.x * w) + 8
        text_y = int(lm.y * h) - 8

        text = f"{angle_value:.0f}"

        cv2.putText(
            image, text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, font_thickness
        )

    return image


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    text_color: Tuple[int, int, int] = FPS_TEXT_COLOR,
) -> np.ndarray:
    """
    在图像上绘制 FPS 信息

    Args:
        image: BGR 格式图像
        fps: 帧率值
        position: 文本位置
        text_color: 文本颜色

    Returns:
        标注后的图像
    """
    cv2.putText(
        image, f"FPS: {fps:.1f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        FPS_FONT_SCALE, text_color, FPS_FONT_THICKNESS
    )
    return image


def run_realtime_demo(
    estimator,
    device_id: int = 0,
    width: int = 640,
    height: int = 480,
    show_angles: bool = True,
    window_name: str = "Pose Estimation Demo",
):
    """
    打开摄像头并实时显示骨骼标注视频

    Args:
        estimator: PoseEstimator 实例
        device_id: 摄像头设备 ID
        width: 画面宽度
        height: 画面高度
        show_angles: 是否显示关节角度
        window_name: 窗口名称

    按 'q' 或 ESC 退出。
    """
    from .video_source import CameraSource

    print(f"正在打开摄像头 (device_id={device_id})...")
    print("按 'q' 或 ESC 退出")

    with CameraSource(device_id, width=width, height=height) as source:
        prev_time = time.time()
        fps = 0.0

        while True:
            success, frame = source.read()
            if not success:
                print("无法读取摄像头画面")
                break

            # 姿态估计
            pose_frame = estimator.estimate_frame(frame)

            # 绘制骨骼图
            if pose_frame is not None:
                draw_skeleton(frame, pose_frame)

                if show_angles:
                    draw_angles(frame, pose_frame)

            # 计算并绘制 FPS
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 0.7 * fps + 0.3 * (1.0 / dt)  # 平滑FPS
            prev_time = current_time
            draw_fps(frame, fps)

            # 显示
            cv2.imshow(window_name, frame)

            # 按键检测
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q 或 ESC
                break

    cv2.destroyAllWindows()
    print("实时演示已结束")
