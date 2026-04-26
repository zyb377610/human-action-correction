"""
姿态估计模块

基于 MediaPipe 的人体姿态估计，提供关键点提取、特征工程和可视化能力。

主要组件:
    - PoseEstimator: 姿态估计核心类
    - PoseFrame / PoseSequence: 数据结构
    - FeatureExtractor: 特征工程
    - VideoSource / CameraSource / FileSource: 视频输入
    - draw_skeleton / draw_angles: 可视化工具
"""

from .data_types import (
    PoseLandmark,
    PoseFrame,
    PoseSequence,
    NUM_LANDMARKS,
    LANDMARK_NAMES,
    POSE_CONNECTIONS,
)
from .estimator import PoseEstimator
from .feature_extractor import (
    FeatureExtractor,
    calculate_angle,
    get_joint_angles,
    get_bone_lengths,
    get_normalized_bone_ratios,
    calculate_velocity,
    calculate_sequence_velocity,
    JOINT_ANGLE_DEFINITIONS,
    BONE_DEFINITIONS,
)
from .video_source import (
    VideoSource,
    CameraSource,
    FileSource,
)
from .visualizer import (
    draw_skeleton,
    draw_angles,
    draw_fps,
    run_realtime_demo,
)

__all__ = [
    # 数据类型
    "PoseLandmark",
    "PoseFrame",
    "PoseSequence",
    "NUM_LANDMARKS",
    "LANDMARK_NAMES",
    "POSE_CONNECTIONS",
    # 估计器
    "PoseEstimator",
    # 特征工程
    "FeatureExtractor",
    "calculate_angle",
    "get_joint_angles",
    "get_bone_lengths",
    "get_normalized_bone_ratios",
    "calculate_velocity",
    "calculate_sequence_velocity",
    "JOINT_ANGLE_DEFINITIONS",
    "BONE_DEFINITIONS",
    # 视频源
    "VideoSource",
    "CameraSource",
    "FileSource",
    # 可视化
    "draw_skeleton",
    "draw_angles",
    "draw_fps",
    "run_realtime_demo",
]