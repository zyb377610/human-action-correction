"""
关节角度计算工具

基于三点向量夹角计算人体关节角度，
支持用户动作与标准模板的角度对比。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose_estimation.data_types import PoseSequence, LANDMARK_NAMES

logger = logging.getLogger(__name__)


# 关键角度定义：{角度名称: (点A索引, 顶点B索引, 点C索引)}
# 角度 = ∠ABC，即从 BA 到 BC 的夹角
ANGLE_DEFINITIONS = {
    # 膝关节角（髋-膝-踝）
    "left_knee_angle": (23, 25, 27),    # left_hip - left_knee - left_ankle
    "right_knee_angle": (24, 26, 28),   # right_hip - right_knee - right_ankle
    # 肘关节角（肩-肘-腕）
    "left_elbow_angle": (11, 13, 15),   # left_shoulder - left_elbow - left_wrist
    "right_elbow_angle": (12, 14, 16),  # right_shoulder - right_elbow - right_wrist
    # 髋关节角（肩-髋-膝）
    "left_hip_angle": (11, 23, 25),     # left_shoulder - left_hip - left_knee
    "right_hip_angle": (12, 24, 26),    # right_shoulder - right_hip - right_knee
    # 肩关节角（髋-肩-肘）
    "left_shoulder_angle": (23, 11, 13),  # left_hip - left_shoulder - left_elbow
    "right_shoulder_angle": (24, 12, 14), # right_hip - right_shoulder - right_elbow
    # 躯干侧倾角（肩中点-髋中点-垂直参考）— 用 左肩-左髋-右髋 近似
    "trunk_angle": (11, 23, 24),        # left_shoulder - left_hip - right_hip
}

# 角度中文名称映射
ANGLE_DISPLAY_NAMES = {
    "left_knee_angle": "左膝关节角",
    "right_knee_angle": "右膝关节角",
    "left_elbow_angle": "左肘关节角",
    "right_elbow_angle": "右肘关节角",
    "left_hip_angle": "左髋关节角",
    "right_hip_angle": "右髋关节角",
    "left_shoulder_angle": "左肩关节角",
    "right_shoulder_angle": "右肩关节角",
    "trunk_angle": "躯干角",
}


def compute_angle_3d(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    """
    计算三点构成的角度 ∠ABC（3D 向量夹角）

    Args:
        point_a: 点 A 坐标 (3,)
        point_b: 顶点 B 坐标 (3,)
        point_c: 点 C 坐标 (3,)

    Returns:
        角度值（度），范围 [0, 180]
    """
    ba = point_a - point_b
    bc = point_c - point_b

    # 向量模长
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    # 余弦值
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


class AngleCalculator:
    """
    关节角度计算器

    计算人体关键角度（膝角、肘角、髋角等），
    支持单帧计算、序列平均和用户-模板角度对比。

    使用示例:
        calc = AngleCalculator()
        angles = calc.compute_sequence_angles(sequence)
        deviations = calc.compare_angles(user_seq, template_seq, path)
    """

    def __init__(
        self,
        angle_definitions: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        """
        Args:
            angle_definitions: 自定义角度定义，None 使用默认定义
        """
        self._definitions = angle_definitions or ANGLE_DEFINITIONS

    def compute_frame_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        计算单帧所有关节角度

        Args:
            landmarks: 关键点数组 (33, 4)，列为 [x, y, z, visibility]

        Returns:
            {角度名称: 角度值（度）}
        """
        angles = {}
        for name, (idx_a, idx_b, idx_c) in self._definitions.items():
            # 使用 xyz 三维坐标
            point_a = landmarks[idx_a, :3]
            point_b = landmarks[idx_b, :3]
            point_c = landmarks[idx_c, :3]
            angles[name] = compute_angle_3d(point_a, point_b, point_c)
        return angles

    def compute_sequence_angles(
        self, sequence: PoseSequence
    ) -> Dict[str, float]:
        """
        计算序列的平均关节角度

        Args:
            sequence: 姿态序列

        Returns:
            {角度名称: 平均角度值（度）}
        """
        arr = sequence.to_numpy()  # (T, 33, 4)
        if arr.shape[0] == 0:
            return {name: 0.0 for name in self._definitions}

        all_angles = {name: [] for name in self._definitions}

        for t in range(arr.shape[0]):
            frame_angles = self.compute_frame_angles(arr[t])
            for name, angle in frame_angles.items():
                all_angles[name].append(angle)

        return {name: float(np.mean(vals)) for name, vals in all_angles.items()}

    def compare_angles(
        self,
        user_seq: PoseSequence,
        template_seq: PoseSequence,
        path: List[Tuple[int, int]],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        沿 DTW 对齐路径对比用户和模板的关节角度

        Args:
            user_seq: 用户动作序列
            template_seq: 标准模板序列
            path: DTW 对齐路径 [(i, j), ...]

        Returns:
            {角度名称: (用户平均角度, 模板平均角度, 角度差)}
            角度差 = 用户角度 - 模板角度（正值表示用户角度更大）
        """
        user_arr = user_seq.to_numpy()    # (N, 33, 4)
        tmpl_arr = template_seq.to_numpy()  # (M, 33, 4)
        n_user = len(user_arr)
        n_tmpl = len(tmpl_arr)

        angle_diffs = {name: [] for name in self._definitions}
        user_angles_all = {name: [] for name in self._definitions}
        tmpl_angles_all = {name: [] for name in self._definitions}

        for i, j in path:
            # path 索引可能来自预处理（重采样）后的序列，超过原始序列长度时夹紧
            ii = min(int(i), n_user - 1) if n_user > 0 else 0
            jj = min(int(j), n_tmpl - 1) if n_tmpl > 0 else 0
            if n_user == 0 or n_tmpl == 0:
                continue
            user_angles = self.compute_frame_angles(user_arr[ii])
            tmpl_angles = self.compute_frame_angles(tmpl_arr[jj])

            for name in self._definitions:
                u_a = user_angles[name]
                t_a = tmpl_angles[name]
                user_angles_all[name].append(u_a)
                tmpl_angles_all[name].append(t_a)
                angle_diffs[name].append(u_a - t_a)

        result = {}
        for name in self._definitions:
            if user_angles_all[name]:
                user_mean = float(np.nanmean(user_angles_all[name]))
                tmpl_mean = float(np.nanmean(tmpl_angles_all[name]))
                diff_mean = float(np.nanmean(angle_diffs[name]))
            else:
                user_mean = tmpl_mean = diff_mean = 0.0
            result[name] = (user_mean, tmpl_mean, diff_mean)

        return result

    @staticmethod
    def get_display_name(angle_name: str) -> str:
        """获取角度的中文显示名称"""
        return ANGLE_DISPLAY_NAMES.get(angle_name, angle_name)