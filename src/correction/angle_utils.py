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

# 角度可见度阈值：构成角度的三个关键点中，任一 visibility 低于此值则该角度无效
ANGLE_VISIBILITY_THRESHOLD = 0.3
# 角度帧占比阈值：角度在序列中至少该比例的帧上有效，才视为"整体有效"
ANGLE_PRESENCE_RATIO = 0.6


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


def compute_valid_angles(
    sequence,
    visibility_threshold: float = ANGLE_VISIBILITY_THRESHOLD,
    presence_ratio_threshold: float = ANGLE_PRESENCE_RATIO,
    valid_joint_indices: Optional[List[int]] = None,
) -> List[str]:
    """
    从模板序列中提取"有效角度集"（模板权威性原则）。

    判定策略（两级）：
    1. 若提供了 valid_joint_indices（来自 compute_valid_joints），
       则以"角度顶点（中间关节点 B）是否在有效关节集中"为核心判据。
       角度测量的是关节 B 处的弯曲程度，只要 B 本身可见，
       即使参考点 A/C 不完全可见，该角度仍具有参考价值。
       （例如：肩角 ∠(髋-肩-肘) 的顶点是肩，肩可见则肩角有效，
        即使髋不在画面中，其推断位置仍能提供躯干方向参考。）
    2. 若未提供 valid_joint_indices，则回退到基于三点的可见度检查。

    Args:
        sequence: PoseSequence（通常是模板序列）
        visibility_threshold: 单帧视为"可见"的置信度阈值（仅回退模式使用）
        presence_ratio_threshold: 角度视为"整体有效"的帧占比阈值（仅回退模式使用）
        valid_joint_indices: 有效关节索引列表（来自 compute_valid_joints），
                             None 则使用回退模式

    Returns:
        有效角度名称列表（保持 ANGLE_DEFINITIONS 中的顺序）
    """
    arr = sequence.to_numpy()  # (T, 33, 4)
    T = arr.shape[0]
    if T == 0:
        return list(ANGLE_DEFINITIONS.keys())

    # === 模式1：基于关节有效集（顶点判据） ===
    if valid_joint_indices is not None:
        valid_set = set(valid_joint_indices)
        result = []
        for name, (idx_a, idx_b, idx_c) in ANGLE_DEFINITIONS.items():
            # 判据：顶点 B（角度测量位置）必须在有效关节集中
            if idx_b not in valid_set:
                continue
            # 额外安全检查：顶点 B 的坐标数据必须存在
            if idx_b >= arr.shape[1]:
                continue
            # 顶点的平均可见度至少 > 0（确保不是纯噪声）
            avg_vis_b = float(np.mean(arr[:, idx_b, 3]))
            if avg_vis_b < 0.01:
                continue
            result.append(name)

        if len(result) < 2:
            return list(ANGLE_DEFINITIONS.keys())
        return result

    # === 模式2：回退 — 三点可见度检查 ===
    valid = []
    for name, (idx_a, idx_b, idx_c) in ANGLE_DEFINITIONS.items():
        if idx_a >= arr.shape[1] or idx_b >= arr.shape[1] or idx_c >= arr.shape[1]:
            continue
        vis_a = arr[:, idx_a, 3]
        vis_b = arr[:, idx_b, 3]
        vis_c = arr[:, idx_c, 3]
        # 三个关键点全部可见的帧占比
        all_visible = (vis_a >= visibility_threshold) & \
                      (vis_b >= visibility_threshold) & \
                      (vis_c >= visibility_threshold)
        presence = float(np.mean(all_visible))
        if presence >= presence_ratio_threshold:
            valid.append(name)

    # 极端情况：若阈值太严导致全部被剔，退回全角度集
    if len(valid) < 2:
        return list(ANGLE_DEFINITIONS.keys())
    return valid


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
        valid_angle_names: Optional[List[str]] = None,
        visibility_threshold: Optional[float] = None,
    ):
        """
        Args:
            angle_definitions: 自定义角度定义，None 使用默认定义
            valid_angle_names: 有效角度名称列表，None 使用全部定义角度；
                              传入后仅计算列表中包含的角度
            visibility_threshold: 角度可见度阈值，None 不检查可见度；
                                 设置后，构成角度的任一关键点 visibility
                                 低于阈值则该角度返回 NaN
        """
        self._definitions = angle_definitions or ANGLE_DEFINITIONS
        self._valid_names = valid_angle_names  # None = 全部有效
        self._vis_threshold = visibility_threshold

    @property
    def valid_angle_names(self) -> List[str]:
        """当前有效的角度名称列表"""
        if self._valid_names is not None:
            return [n for n in self._valid_names if n in self._definitions]
        return list(self._definitions.keys())

    def compute_frame_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        计算单帧所有关节角度（仅计算有效角度）

        若设置了 visibility_threshold，则对构成角度的三个关键点
        逐一检查可见度：任一 visibility < threshold → 该角度返回 NaN。

        Args:
            landmarks: 关键点数组 (33, 4)，列为 [x, y, z, visibility]

        Returns:
            {角度名称: 角度值（度）}，无效角度值为 NaN
        """
        angles = {}
        names_to_compute = self._valid_names if self._valid_names is not None else list(self._definitions.keys())

        for name in names_to_compute:
            if name not in self._definitions:
                continue
            idx_a, idx_b, idx_c = self._definitions[name]

            # 可见度门控：任一关键点不可见 → NaN
            if self._vis_threshold is not None:
                vis_a = float(landmarks[idx_a, 3])
                vis_b = float(landmarks[idx_b, 3])
                vis_c = float(landmarks[idx_c, 3])
                if min(vis_a, vis_b, vis_c) < self._vis_threshold:
                    angles[name] = float('nan')
                    continue

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
        计算序列的平均关节角度（仅有效角度，跳过 NaN）

        Args:
            sequence: 姿态序列

        Returns:
            {角度名称: 平均角度值（度）}
        """
        arr = sequence.to_numpy()  # (T, 33, 4)
        valid_names = self._valid_names if self._valid_names is not None else list(self._definitions.keys())
        if arr.shape[0] == 0:
            return {name: 0.0 for name in valid_names}

        all_angles = {name: [] for name in valid_names}

        for t in range(arr.shape[0]):
            frame_angles = self.compute_frame_angles(arr[t])
            for name in valid_names:
                val = frame_angles.get(name, float('nan'))
                if not np.isnan(val):
                    all_angles[name].append(val)

        return {name: float(np.mean(vals)) if vals else 0.0
                for name, vals in all_angles.items()}

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