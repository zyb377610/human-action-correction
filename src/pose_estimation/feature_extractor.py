"""
姿态特征工程模块

基于骨骼关键点原始坐标计算衍生特征：
- 关节角度（三点向量夹角）
- 骨骼长度与归一化比例
- 运动速度与加速度
- 综合特征向量
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_types import PoseFrame, PoseSequence, LANDMARK_NAMES, NUM_LANDMARKS

logger = logging.getLogger(__name__)


# ===== 核心关节角度定义 =====
# 格式: (关节名称, 起点索引, 中间点索引, 终点索引)
# 角度 = 起点-中间点-终点 的夹角
JOINT_ANGLE_DEFINITIONS = {
    "left_elbow":    (11, 13, 15),   # 左肩-左肘-左腕
    "right_elbow":   (12, 14, 16),   # 右肩-右肘-右腕
    "left_shoulder":  (13, 11, 23),   # 左肘-左肩-左髋
    "right_shoulder": (14, 12, 24),   # 右肘-右肩-右髋
    "left_hip":      (11, 23, 25),   # 左肩-左髋-左膝
    "right_hip":     (12, 24, 26),   # 右肩-右髋-右膝
    "left_knee":     (23, 25, 27),   # 左髋-左膝-左踝
    "right_knee":    (24, 26, 28),   # 右髋-右膝-右踝
}

# ===== 骨骼段定义 =====
# 格式: (骨骼名称, 起点索引, 终点索引)
BONE_DEFINITIONS = {
    "left_upper_arm":  (11, 13),    # 左上臂: 左肩-左肘
    "left_forearm":    (13, 15),    # 左前臂: 左肘-左腕
    "right_upper_arm": (12, 14),    # 右上臂: 右肩-右肘
    "right_forearm":   (14, 16),    # 右前臂: 右肘-右腕
    "left_thigh":      (23, 25),    # 左大腿: 左髋-左膝
    "left_shin":       (25, 27),    # 左小腿: 左膝-左踝
    "right_thigh":     (24, 26),    # 右大腿: 右髋-右膝
    "right_shin":      (26, 28),    # 右小腿: 右膝-右踝
    "shoulders":       (11, 12),    # 双肩宽
    "hips":            (23, 24),    # 双髋宽
    "left_torso":      (11, 23),    # 左躯干: 左肩-左髋
    "right_torso":     (12, 24),    # 右躯干: 右肩-右髋
}

# 可见度阈值
VISIBILITY_THRESHOLD = 0.5


def calculate_angle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """
    计算三个点构成的角度（以 p2 为顶点）

    Args:
        p1: 起点坐标 (x, y[, z])
        p2: 顶点坐标 (x, y[, z])
        p3: 终点坐标 (x, y[, z])

    Returns:
        角度值（角度制，[0, 180]），无法计算时返回 NaN
    """
    v1 = p1 - p2
    v2 = p3 - p2

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return float("nan")

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    # 数值稳定性：clamp 到 [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def get_joint_angles(
    frame: PoseFrame,
    use_3d: bool = False,
) -> Dict[str, float]:
    """
    计算单帧的所有核心关节角度

    Args:
        frame: 单帧姿态数据
        use_3d: 是否使用 3D 坐标 (x, y, z)，False 时仅用 (x, y)

    Returns:
        关节角度字典，键为关节名称，值为角度（度）。
        可见度不足的关节标记为 NaN。
    """
    landmarks = frame.landmarks
    angles = {}

    for name, (i1, i2, i3) in JOINT_ANGLE_DEFINITIONS.items():
        lm1, lm2, lm3 = landmarks[i1], landmarks[i2], landmarks[i3]

        # 可见度检查
        if (lm1.visibility < VISIBILITY_THRESHOLD or
            lm2.visibility < VISIBILITY_THRESHOLD or
            lm3.visibility < VISIBILITY_THRESHOLD):
            angles[name] = float("nan")
            logger.debug(f"关节 {name} 的关键点可见度不足，标记为 NaN")
            continue

        if use_3d:
            p1 = np.array([lm1.x, lm1.y, lm1.z])
            p2 = np.array([lm2.x, lm2.y, lm2.z])
            p3 = np.array([lm3.x, lm3.y, lm3.z])
        else:
            p1 = np.array([lm1.x, lm1.y])
            p2 = np.array([lm2.x, lm2.y])
            p3 = np.array([lm3.x, lm3.y])

        angles[name] = calculate_angle(p1, p2, p3)

    return angles


def get_bone_lengths(frame: PoseFrame) -> Dict[str, float]:
    """
    计算单帧的骨骼段长度

    Args:
        frame: 单帧姿态数据

    Returns:
        骨骼长度字典，键为骨骼名称，值为欧氏距离
    """
    landmarks = frame.landmarks
    lengths = {}

    for name, (i1, i2) in BONE_DEFINITIONS.items():
        lm1, lm2 = landmarks[i1], landmarks[i2]
        p1 = np.array([lm1.x, lm1.y])
        p2 = np.array([lm2.x, lm2.y])
        lengths[name] = float(np.linalg.norm(p2 - p1))

    return lengths


def get_normalized_bone_ratios(frame: PoseFrame) -> Dict[str, float]:
    """
    计算归一化骨骼比例

    以躯干长度（肩中点到髋中点）为基准进行归一化。

    Args:
        frame: 单帧姿态数据

    Returns:
        归一化骨骼比例字典
    """
    landmarks = frame.landmarks

    # 计算躯干长度 = 肩中点到髋中点的距离
    shoulder_mid = np.array([
        (landmarks[11].x + landmarks[12].x) / 2,
        (landmarks[11].y + landmarks[12].y) / 2,
    ])
    hip_mid = np.array([
        (landmarks[23].x + landmarks[24].x) / 2,
        (landmarks[23].y + landmarks[24].y) / 2,
    ])
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))

    if torso_length < 1e-8:
        # 躯干长度几乎为零，无法归一化
        return {name: float("nan") for name in BONE_DEFINITIONS}

    lengths = get_bone_lengths(frame)
    ratios = {name: length / torso_length for name, length in lengths.items()}

    return ratios


def calculate_velocity(
    frame1: PoseFrame,
    frame2: PoseFrame,
) -> np.ndarray:
    """
    计算两帧之间各关键点的瞬时速度

    Args:
        frame1: 前一帧
        frame2: 后一帧

    Returns:
        形状为 (33,) 的速度数组（单位：归一化坐标/秒）
    """
    dt = frame2.timestamp - frame1.timestamp
    if abs(dt) < 1e-8:
        return np.zeros(NUM_LANDMARKS, dtype=np.float32)

    arr1 = frame1.to_numpy()[:, :2]  # (33, 2) 取 x, y
    arr2 = frame2.to_numpy()[:, :2]

    displacement = np.linalg.norm(arr2 - arr1, axis=1)  # (33,)
    velocity = displacement / dt

    return velocity.astype(np.float32)


def calculate_sequence_velocity(sequence: PoseSequence) -> np.ndarray:
    """
    计算序列中所有帧的运动速度

    Args:
        sequence: 姿态序列

    Returns:
        形状为 (T-1, 33) 的速度数组，T < 2 时返回空数组
    """
    if sequence.num_frames < 2:
        return np.empty((0, NUM_LANDMARKS), dtype=np.float32)

    velocities = []
    for i in range(1, sequence.num_frames):
        v = calculate_velocity(sequence.frames[i - 1], sequence.frames[i])
        velocities.append(v)

    return np.stack(velocities, axis=0)


class FeatureExtractor:
    """
    综合特征提取器

    将关节角度、骨骼比例和运动速度组合为统一的特征向量。

    使用示例:
        extractor = FeatureExtractor()
        features = extractor.extract_frame_features(pose_frame)
        feature_matrix = extractor.extract_sequence_features(pose_sequence)
    """

    def __init__(self, use_3d: bool = False):
        """
        Args:
            use_3d: 角度计算是否使用 3D 坐标
        """
        self._use_3d = use_3d

        # 特征名称列表（固定顺序）
        self._angle_names = sorted(JOINT_ANGLE_DEFINITIONS.keys())
        self._bone_names = sorted(BONE_DEFINITIONS.keys())

    @property
    def angle_dim(self) -> int:
        """关节角度特征维度"""
        return len(self._angle_names)

    @property
    def bone_ratio_dim(self) -> int:
        """骨骼比例特征维度"""
        return len(self._bone_names)

    @property
    def feature_dim(self) -> int:
        """单帧总特征维度（角度 + 骨骼比例）"""
        return self.angle_dim + self.bone_ratio_dim

    @property
    def feature_names(self) -> List[str]:
        """特征名称列表"""
        return (
            [f"angle_{n}" for n in self._angle_names] +
            [f"ratio_{n}" for n in self._bone_names]
        )

    def extract_frame_features(self, frame: PoseFrame) -> np.ndarray:
        """
        提取单帧特征向量

        Args:
            frame: 单帧姿态数据

        Returns:
            一维特征向量，维度为 feature_dim
        """
        # 关节角度
        angles = get_joint_angles(frame, use_3d=self._use_3d)
        angle_values = [angles[n] for n in self._angle_names]

        # 骨骼比例
        ratios = get_normalized_bone_ratios(frame)
        ratio_values = [ratios[n] for n in self._bone_names]

        features = np.array(angle_values + ratio_values, dtype=np.float32)
        return features

    def extract_sequence_features(
        self,
        sequence: PoseSequence,
    ) -> np.ndarray:
        """
        提取序列特征矩阵

        Args:
            sequence: 姿态序列

        Returns:
            形状为 (T, D) 的特征矩阵，T 为帧数，D 为特征维度
        """
        if sequence.num_frames == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        features = []
        for frame in sequence.frames:
            f = self.extract_frame_features(frame)
            features.append(f)

        return np.stack(features, axis=0)
