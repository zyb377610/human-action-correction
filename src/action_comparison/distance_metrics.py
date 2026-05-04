"""
距离度量函数

支持欧氏距离、余弦距离、曼哈顿距离，提供统一接口。
"""

from typing import Callable, Optional, Sequence

import numpy as np


# ===== 核心关节点索引 =====
# 从 MediaPipe 33 个关键点中筛选 17 个关键骨骼点用于 DTW 比对。
# 包含：头部参考点(1) + 上肢(6) + 躯干(4) + 下肢(6) = 17 个
# 去掉眼睛(1-4)、耳朵(7-8)、嘴角(9-10)、小指/拇指(19-22)等
# 高抖动低信息量的点，保留运动分析核心骨骼。
CORE_JOINT_INDICES = [
    0,   # nose          — 头部朝向参考
    11,  # left_shoulder  — 左肩
    12,  # right_shoulder — 右肩
    13,  # left_elbow     — 左肘
    14,  # right_elbow    — 右肘
    15,  # left_wrist     — 左腕
    16,  # right_wrist    — 右腕
    23,  # left_hip       — 左髋
    24,  # right_hip      — 右髋
    25,  # left_knee      — 左膝
    26,  # right_knee     — 右膝
    27,  # left_ankle     — 左踝
    28,  # right_ankle    — 右踝
    29,  # left_heel      — 左脚跟
    30,  # right_heel     — 右脚跟
    31,  # left_foot_index  — 左脚尖
    32,  # right_foot_index — 右脚尖
]

# 核心关节点中文名称
CORE_JOINT_NAMES = {
    0: "鼻子",
    11: "左肩", 12: "右肩",
    13: "左肘", 14: "右肘",
    15: "左腕", 16: "右腕",
    23: "左髋", 24: "右髋",
    25: "左膝", 26: "右膝",
    27: "左踝", 28: "右踝",
    29: "左脚跟", 30: "右脚跟",
    31: "左脚尖", 32: "右脚尖",
}


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦距离 = 1 - cosine_similarity

    范围 [0, 2]，0 表示方向完全一致
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0  # 零向量视为最大不相似
    return float(1.0 - dot / (norm_a * norm_b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """曼哈顿距离（L1 范数）"""
    return float(np.sum(np.abs(a - b)))


# ===== 统一接口 =====

_METRICS = {
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "manhattan": manhattan_distance,
}


def get_distance_func(name: str) -> Callable:
    """
    获取距离度量函数

    Args:
        name: 度量名称 — "euclidean" / "cosine" / "manhattan"

    Returns:
        距离函数 (a, b) -> float
    """
    name = name.lower().strip()
    if name not in _METRICS:
        raise ValueError(
            f"不支持的距离度量: '{name}'，可选: {list(_METRICS.keys())}"
        )
    return _METRICS[name]


def sequence_to_feature_matrix(
    sequence,
    joint_indices: Optional[Sequence[int]] = None,
    normalize_body_scale: bool = True,
    center_normalize: bool = True,
) -> np.ndarray:
    """
    将 PoseSequence 转换为角度特征矩阵

    使用关节角度作为主特征：角度天然不受位置、视角、体型影响，
    是区分不同动作最本质的特征。

    Args:
        sequence: PoseSequence 对象
        joint_indices: （保留接口兼容，内部不使用）
        normalize_body_scale: （保留接口兼容）
        center_normalize: （保留接口兼容）

    Returns:
        (T, num_angles) ndarray，每个元素是角度值 / 180（归一化到 [0,1]）
    """
    from src.correction.angle_utils import AngleCalculator, ANGLE_DEFINITIONS

    calc = AngleCalculator()
    T = sequence.num_frames
    angle_names = list(ANGLE_DEFINITIONS.keys())
    n_angles = len(angle_names)
    matrix = np.zeros((T, n_angles), dtype=np.float64)

    arr = sequence.to_numpy()  # (T, 33, 4)
    for t in range(T):
        angles = calc.compute_frame_angles(arr[t])
        for k, name in enumerate(angle_names):
            matrix[t, k] = angles[name] / 180.0  # 归一化到 [0,1]

    return matrix


def sequence_to_landmark_matrix(
    sequence,
    joint_indices: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    将 PoseSequence 转换为 **关节坐标** 特征矩阵（用于空间偏差分析）

    与 sequence_to_feature_matrix（角度）不同，本函数返回每帧指定关节的
    xyz 坐标，经过髋中点平移 + 躯干长度缩放归一化，使得不同相机位置、
    不同身高体型的用户/模板可以直接在同一坐标系下做欧氏距离对比。

    Args:
        sequence: PoseSequence 对象
        joint_indices: 关节索引列表，None 使用 CORE_JOINT_INDICES
        normalize: 是否进行身体尺度归一化

    Returns:
        (T, J*3) ndarray，每一行是 J 个关节的 xyz 拼接，已归一化。
    """
    if joint_indices is None:
        joint_indices = CORE_JOINT_INDICES
    joint_indices = list(joint_indices)
    J = len(joint_indices)

    arr = sequence.to_numpy()  # (T, 33, 4)
    T = arr.shape[0]
    matrix = np.zeros((T, J * 3), dtype=np.float64)

    for t in range(T):
        lm = arr[t]  # (33, 4)
        if normalize and lm.shape[0] > 24:
            # 髋中点作为坐标原点
            cx = (lm[23, 0] + lm[24, 0]) / 2.0
            cy = (lm[23, 1] + lm[24, 1]) / 2.0
            cz = (lm[23, 2] + lm[24, 2]) / 2.0
            # 肩中点
            sx = (lm[11, 0] + lm[12, 0]) / 2.0
            sy = (lm[11, 1] + lm[12, 1]) / 2.0
            sz = (lm[11, 2] + lm[12, 2]) / 2.0
            torso = np.sqrt(
                (sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2
            )
            if torso < 1e-3:
                torso = 1.0
        else:
            cx = cy = cz = 0.0
            torso = 1.0

        for k, j in enumerate(joint_indices):
            if j < lm.shape[0]:
                matrix[t, k * 3 + 0] = (lm[j, 0] - cx) / torso
                matrix[t, k * 3 + 1] = (lm[j, 1] - cy) / torso
                matrix[t, k * 3 + 2] = (lm[j, 2] - cz) / torso

    return matrix


def _compute_body_scale(sequence) -> float:
    """计算身体尺度因子（平均躯干长度），用于归一化体型差异"""
    # 肩中点(11,12) → 髋中点(23,24)
    total_torso = 0.0
    count = 0
    for frame in sequence.frames:
        landmarks = frame.landmarks
        if len(landmarks) <= 24:
            continue
        # 肩中点
        sx = (landmarks[11].x + landmarks[12].x) / 2
        sy = (landmarks[11].y + landmarks[12].y) / 2
        # 髋中点
        hx = (landmarks[23].x + landmarks[24].x) / 2
        hy = (landmarks[23].y + landmarks[24].y) / 2
        torso = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        if torso > 0.01:
            total_torso += torso
            count += 1
    return total_torso / count if count > 0 else 1.0


def compute_valid_joints(
    sequence,
    visibility_threshold: float = 0.3,
    presence_ratio_threshold: float = 0.6,
) -> list:
    """
    从模板序列中提取"有效关节集"。

    一个关节被认为"有效"，当且仅当其在至少 `presence_ratio_threshold`
    比例的帧上 visibility >= `visibility_threshold`。

    这为"模板权威性原则"提供支撑：模板里看不到的关节（例如只录了上半身），
    下游对比与矫正也应自动忽略。

    Args:
        sequence: PoseSequence
        visibility_threshold: 单帧视为"可见"的置信度阈值
        presence_ratio_threshold: 关节视为"整体可见"的帧占比阈值

    Returns:
        有效关节索引列表（取自 CORE_JOINT_INDICES 的子集），保证顺序。
    """
    arr = sequence.to_numpy()  # (T, 33, 4)
    T = arr.shape[0]
    if T == 0:
        return list(CORE_JOINT_INDICES)

    valid = []
    for j in CORE_JOINT_INDICES:
        if j >= arr.shape[1]:
            continue
        vis = arr[:, j, 3]
        present = float(np.mean(vis >= visibility_threshold))
        if present >= presence_ratio_threshold:
            valid.append(j)

    # 极端情况：若阈值太严导致全部被剔，退回全核心集
    if len(valid) < 3:
        return list(CORE_JOINT_INDICES)
    return valid


def _rotation_from_shoulder_axis(lm: np.ndarray) -> np.ndarray:
    """
    从单帧 landmarks 计算一个 2D 旋转矩阵，使左→右肩向量对齐到 +x 方向。

    只做 xy 平面的旋转（大多数摄像场景下肩轴倾斜主要体现在 xy）。

    Args:
        lm: (33, 4) 原始关键点

    Returns:
        2x2 旋转矩阵
    """
    if lm.shape[0] <= 12:
        return np.eye(2)
    lsh = lm[11, :2]
    rsh = lm[12, :2]
    axis = rsh - lsh
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return np.eye(2)
    axis = axis / n
    # axis = (cos θ, sin θ) 希望旋转到 (1, 0) → 旋转矩阵为 [[cos, sin], [-sin, cos]]
    c, s = axis[0], axis[1]
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def sequence_to_landmark_matrix_masked(
    sequence,
    joint_indices: Sequence[int],
    align_shoulder: bool = True,
) -> np.ndarray:
    """
    增强版坐标矩阵提取：
    - 仅保留指定关节（joint_indices）
    - 髋中点平移归一化 + 躯干长度缩放归一化
    - 可选：基于肩轴的 2D 旋转对齐，进一步抵消不同摄像角度下上身倾斜

    Args:
        sequence: PoseSequence
        joint_indices: 要保留的关节索引列表
        align_shoulder: 是否按肩轴对齐（推荐 True，消除左右镜像/侧身差异）

    Returns:
        (T, J*3) ndarray
    """
    joint_indices = list(joint_indices)
    J = len(joint_indices)
    arr = sequence.to_numpy()  # (T, 33, 4)
    T = arr.shape[0]
    matrix = np.zeros((T, J * 3), dtype=np.float64)

    for t in range(T):
        lm = arr[t]
        if lm.shape[0] > 24:
            cx = (lm[23, 0] + lm[24, 0]) / 2.0
            cy = (lm[23, 1] + lm[24, 1]) / 2.0
            cz = (lm[23, 2] + lm[24, 2]) / 2.0
            sx = (lm[11, 0] + lm[12, 0]) / 2.0
            sy = (lm[11, 1] + lm[12, 1]) / 2.0
            sz = (lm[11, 2] + lm[12, 2]) / 2.0
            torso = np.sqrt(
                (sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2
            )
            if torso < 1e-3:
                torso = 1.0
        else:
            cx = cy = cz = 0.0
            torso = 1.0

        R = _rotation_from_shoulder_axis(lm) if align_shoulder else np.eye(2)

        for k, j in enumerate(joint_indices):
            if j < lm.shape[0]:
                dx = (lm[j, 0] - cx) / torso
                dy = (lm[j, 1] - cy) / torso
                dz = (lm[j, 2] - cz) / torso
                # xy 做旋转对齐
                xy = R @ np.array([dx, dy])
                matrix[t, k * 3 + 0] = xy[0]
                matrix[t, k * 3 + 1] = xy[1]
                matrix[t, k * 3 + 2] = dz

    return matrix
