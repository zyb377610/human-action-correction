"""
实时矫正反馈引擎

基于固定窗口 DTW + 关节角度偏差的即时反馈模块。
每帧缓存最近 N 帧 landmarks，与标准模板对应进度的 N 帧做 DTW 对比，
结合角度偏差阈值和规则引擎，生成实时矫正建议。

设计目标：单次 analyze_frame() 调用耗时 ≤10ms（不含 MediaPipe 推理）。
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose_estimation.data_types import PoseSequence
from src.data.preprocessing import filter_skeleton_outliers, extract_action_segment
from src.action_comparison.distance_metrics import (
    sequence_to_feature_matrix,
    get_distance_func,
    CORE_JOINT_INDICES,
    CORE_JOINT_NAMES,
    weighted_euclidean_distance,
    AXIS_WEIGHTS,
    JOINT_IMPORTANCE_WEIGHTS,
)
from src.action_comparison.dtw_algorithms import compute_dtw
from src.correction.angle_utils import AngleCalculator
from src.pose_estimation.data_types import LANDMARK_NAMES

logger = logging.getLogger(__name__)


def _window_to_angle_matrix(
    window: np.ndarray,
    valid_angle_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    将关键点窗口转换为角度特征矩阵

    Args:
        window: (T, 33, 4) 关键点窗口
        valid_angle_names: 有效角度名称列表，None 使用全部定义角度

    Returns:
        (T, num_angles) 角度特征矩阵，值归一化到 [0,1]
    """
    from src.correction.angle_utils import AngleCalculator, ANGLE_DEFINITIONS

    if valid_angle_names is not None:
        angle_names = [n for n in valid_angle_names if n in ANGLE_DEFINITIONS]
        if not angle_names:
            angle_names = list(ANGLE_DEFINITIONS.keys())
    else:
        angle_names = list(ANGLE_DEFINITIONS.keys())

    calc = AngleCalculator(valid_angle_names=angle_names)
    T = window.shape[0]
    n_angles = len(angle_names)
    result = np.zeros((T, n_angles), dtype=np.float64)

    for t in range(T):
        angles = calc.compute_frame_angles(window[t])
        for k, name in enumerate(angle_names):
            val = angles.get(name, float('nan'))
            if np.isnan(val):
                result[t, k] = 0.5  # 中立值
            else:
                result[t, k] = val / 180.0

    return result


# ================================================================
# 数据类
# ================================================================

class Severity(str, Enum):
    """建议严重等级"""
    INFO = "info"          # 轻微偏差
    WARNING = "warning"    # 中等偏差
    ERROR = "error"        # 严重偏差


@dataclass
class FeedbackItem:
    """
    单条实时矫正建议

    Attributes:
        joint_name: 关节名称（如 "left_knee"）
        joint_display: 关节中文名称（如 "左膝"）
        deviation_deg: 角度偏差（度），source="angle" 时有效
        deviation_spatial: 空间坐标偏差（归一化单位），source="spatial" 时有效
        direction: 偏差方向描述（如 "角度过小" / "位置偏右"）
        advice: 矫正提示文字（如 "请将膝盖向外展开"）
        severity: 严重等级
        source: 偏差来源 — "angle"（角度）或 "spatial"（空间坐标）
    """
    joint_name: str = ""
    joint_display: str = ""
    deviation_deg: float = 0.0
    deviation_spatial: float = 0.0
    direction: str = ""
    advice: str = ""
    severity: Severity = Severity.INFO
    source: str = "angle"


@dataclass
class FeedbackSnapshot:
    """
    单帧的实时反馈快照

    Attributes:
        items: 矫正建议列表（角度 + 空间合并）
        has_pose: 是否检测到人体姿态
        window_similarity: 固定窗口 DTW 相似度 [0, 1]
        frame_index: 当前帧索引
        timestamp: 时间戳（秒）
        buffer_size: 当前帧缓冲区大小
        dtw_aligned_template_frame: DTW 对齐后的模板帧索引（-1 表示未计算）
        spatial_deviation_overall: 空间坐标整体平均偏差（归一化单位，-1 表示未计算）
    """
    items: List[FeedbackItem] = field(default_factory=list)
    has_pose: bool = False
    window_similarity: float = 0.0
    frame_index: int = 0
    timestamp: float = 0.0
    buffer_size: int = 0
    dtw_aligned_template_frame: int = -1
    spatial_deviation_overall: float = -1.0

    @property
    def num_issues(self) -> int:
        """需要矫正的问题数量"""
        return len(self.items)

    @property
    def has_issues(self) -> bool:
        return self.num_issues > 0

    def to_markdown(self) -> str:
        """
        转换为 Markdown 格式的建议面板文本

        Returns:
            格式化的 Markdown 字符串
        """
        if not self.has_pose:
            return "⚠️ 未检测到人体姿态，请确保全身在画面中"

        lines = [
            f"**窗口相似度**: {self.window_similarity:.0%}　"
            f"**帧**: {self.frame_index}　"
            f"**缓冲**: {self.buffer_size}",
        ]
        if self.dtw_aligned_template_frame >= 0:
            lines.append(
                f"**DTW对齐模板帧**: {self.dtw_aligned_template_frame}"
            )
        if self.spatial_deviation_overall >= 0:
            level = ("轻微" if self.spatial_deviation_overall < 0.08
                     else "中等" if self.spatial_deviation_overall < 0.18
                     else "较大")
            lines.append(f"**空间偏差**: {self.spatial_deviation_overall:.3f} ({level})")
        lines.append("")

        if not self.has_issues:
            lines.append("✅ **姿态正确，继续保持！**")
        else:
            for item in self.items:
                icon = {
                    Severity.ERROR: "🔴",
                    Severity.WARNING: "🟡",
                    Severity.INFO: "🟢",
                }.get(item.severity, "⚪")
                if item.source == "spatial":
                    lines.append(
                        f"{icon} 📍 **{item.joint_display}** — "
                        f"{item.advice}（偏移 {item.deviation_spatial:.3f}）"
                    )
                else:
                    lines.append(
                        f"{icon} **{item.joint_display}** — "
                        f"{item.advice}（偏差 {item.deviation_deg:.0f}°）"
                    )

        return "\n".join(lines)


# ================================================================
# 关节名称映射
# ================================================================

JOINT_DISPLAY_NAMES = {
    "left_knee": "左膝",
    "right_knee": "右膝",
    "left_hip": "左髋",
    "right_hip": "右髋",
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
}

# 角度偏差方向和建议
ANGLE_ADVICE = {
    "left_knee": {"small": "左膝弯曲不足，请下蹲更深", "large": "左膝弯曲过度，请适当伸展"},
    "right_knee": {"small": "右膝弯曲不足，请下蹲更深", "large": "右膝弯曲过度，请适当伸展"},
    "left_hip": {"small": "左髋角度过小，请挺直腰背", "large": "左髋角度过大，请适当前倾"},
    "right_hip": {"small": "右髋角度过小，请挺直腰背", "large": "右髋角度过大，请适当前倾"},
    "left_elbow": {"small": "左肘弯曲过度，请伸展手臂", "large": "左肘过于伸直，请适当弯曲"},
    "right_elbow": {"small": "右肘弯曲过度，请伸展手臂", "large": "右肘过于伸直，请适当弯曲"},
    "left_shoulder": {"small": "左肩角度不足，请抬高手臂", "large": "左肩角度过大，请降低手臂"},
    "right_shoulder": {"small": "右肩角度不足，请抬高手臂", "large": "右肩角度过大，请降低手臂"},
}


# ================================================================
# 实时反馈引擎
# ================================================================

class RealtimeFeedbackEngine:
    """
    实时矫正反馈引擎

    基于固定窗口 DTW + 关节角度偏差的即时分析。
    每帧调用 analyze_frame()，内部缓存最近 N 帧，
    与模板对应进度的 N 帧做 DTW 对比，同时检查角度偏差。

    使用示例:
        engine = RealtimeFeedbackEngine(template_sequence, algorithm="fastdtw")
        snapshot = engine.analyze_frame(landmarks, frame_idx=10, expected_total=60)
        print(snapshot.to_markdown())
    """

    def __init__(
        self,
        template_sequence: PoseSequence,
        algorithm: str = "fastdtw",
        window_size: int = 10,
        angle_threshold: float = 15.0,
        spatial_threshold: float = 0.06,
        dedup_interval: float = 2.0,
    ):
        """
        Args:
            template_sequence: 标准动作模板序列
            algorithm: DTW 算法名称 (dtw / fastdtw / ddtw)
            window_size: 固定窗口大小（帧数）
            angle_threshold: 角度偏差阈值（度），低于阈值不报告
            spatial_threshold: 空间坐标偏差阈值（归一化单位），低于阈值不报告
            dedup_interval: 建议去重间隔（秒），同一建议在此间隔内不重复
        """
        # 对模板也进行骨骼过滤和片段提取（确保模板干净）
        template_sequence = filter_skeleton_outliers(template_sequence)
        template_sequence = extract_action_segment(template_sequence)

        self._template = template_sequence
        self._template_array = template_sequence.to_numpy()  # (T, 33, 4)
        self._algorithm = algorithm
        self._window_size = window_size
        self._angle_threshold = angle_threshold
        self._spatial_threshold = spatial_threshold
        self._dedup_interval = dedup_interval

        # 模板权威性：确定有效角度集
        from src.correction.angle_utils import compute_valid_angles, ANGLE_DEFINITIONS
        self._valid_angle_names = compute_valid_angles(template_sequence)
        logger.debug(
            f"RealtimeFeedbackEngine 有效角度: "
            f"{len(self._valid_angle_names)}/{len(ANGLE_DEFINITIONS)}"
        )

        # 帧缓冲区
        self._frame_buffer: deque = deque(maxlen=window_size)

        # 角度计算器
        self._angle_calc = AngleCalculator(
            valid_angle_names=self._valid_angle_names
        )

        # 建议去重缓存 {advice_key: last_shown_timestamp}
        self._dedup_cache: Dict[str, float] = {}

        # 帧计数器
        self._frame_count = 0

        # 骨骼一致性检测：记录上一帧的躯干特征，用于检测实时帧中的骨骼突变
        self._last_torso_length: Optional[float] = None
        self._last_body_center: Optional[np.ndarray] = None

        logger.info(
            f"RealtimeFeedbackEngine 初始化: algorithm={algorithm}, "
            f"window={window_size}, threshold={angle_threshold}°, "
            f"template_frames={template_sequence.num_frames}"
        )

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: str):
        self._algorithm = value

    def reset(self):
        """重置引擎状态（新一轮跟做前调用）"""
        self._frame_buffer.clear()
        self._dedup_cache.clear()
        self._frame_count = 0
        self._last_torso_length = None
        self._last_body_center = None

    def _check_skeleton_consistency(self, landmarks: np.ndarray) -> bool:
        """
        检查当前帧骨骼是否与前一帧一致（检测多人遮挡跳变）

        Args:
            landmarks: (33, 4) 当前帧关键点

        Returns:
            True 表示一致（正常），False 表示可能是遮挡或换人
        """
        # 计算当前帧的躯干长度和身体中心
        sx = (landmarks[11, 0] + landmarks[12, 0]) / 2
        sy = (landmarks[11, 1] + landmarks[12, 1]) / 2
        hx = (landmarks[23, 0] + landmarks[24, 0]) / 2
        hy = (landmarks[23, 1] + landmarks[24, 1]) / 2
        torso = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        center = np.array([(sx + hx) / 2, (sy + hy) / 2])

        if torso < 0.01:
            return False  # 躯干长度异常

        is_consistent = True

        if self._last_torso_length is not None:
            # 躯干尺度突变检测
            scale_change = abs(torso - self._last_torso_length) / self._last_torso_length
            if scale_change > 0.4:
                is_consistent = False

        if self._last_body_center is not None:
            # 身体中心跳变检测
            center_jump = np.sqrt(np.sum((center - self._last_body_center) ** 2))
            if center_jump > 0.3:
                is_consistent = False

        # 更新记录（只在一致时更新，避免被错误帧污染）
        if is_consistent:
            self._last_torso_length = torso
            self._last_body_center = center

        return is_consistent

    def analyze_frame(
        self,
        landmarks: np.ndarray,
        frame_index: int,
        expected_total_frames: int,
    ) -> FeedbackSnapshot:
        """
        分析单帧，返回实时反馈快照

        Args:
            landmarks: 当前帧关键点 (33, 4)
            frame_index: 当前帧索引（从 0 开始）
            expected_total_frames: 预期总帧数（用于进度映射）

        Returns:
            FeedbackSnapshot
        """
        self._frame_count += 1
        now = time.time()

        # ======== 骨骼一致性检测 ========
        if not self._check_skeleton_consistency(landmarks):
            logger.debug(f"帧 {frame_index} 骨骼突变，跳过分析")
            return FeedbackSnapshot(
                has_pose=True,
                window_similarity=0.0,
                frame_index=frame_index,
                timestamp=now,
                buffer_size=len(self._frame_buffer),
            )

        # 缓存当前帧
        self._frame_buffer.append(landmarks.copy())
        buffer_size = len(self._frame_buffer)

        # 1. 计算当前帧角度
        user_angles = self._angle_calc.compute_frame_angles(landmarks)

        # 2. 固定窗口 DTW（如果缓冲区足够）
        window_similarity = 0.0
        dtw_path = []
        template_window = np.zeros((0, 33, 4))
        aligned_template_idx = -1
        spatial_items: List[FeedbackItem] = []
        spatial_overall = -1.0

        if buffer_size >= 3:
            window_similarity, dtw_path, user_win, template_window = \
                self._compute_window_dtw(frame_index, expected_total_frames)

            # === 从 DTW 路径中提取当前帧的对齐模板帧 ===
            if dtw_path and user_win.shape[0] > 0 and template_window.shape[0] > 0:
                # 用户窗口的最后一帧索引 = user_win.shape[0] - 1
                user_last = user_win.shape[0] - 1
                # 查找 DTW 路径中所有 user_idx == user_last 的配对
                aligned_tmpl_indices = [
                    j for i, j in dtw_path if i == user_last
                ]
                if aligned_tmpl_indices:
                    # 取中位数（多数投票）
                    aligned_template_idx = int(np.median(aligned_tmpl_indices))
                    # 防御越界
                    aligned_template_idx = min(
                        aligned_template_idx, template_window.shape[0] - 1
                    )

                    # === 空间坐标偏差检测 ===
                    template_lm = template_window[aligned_template_idx]  # (33, 4)
                    spatial_items, spatial_overall = self._detect_spatial_deviations(
                        landmarks, template_lm, now
                    )

        # 3. 确定用于角度对比的模板帧
        #    优先使用 DTW 对齐结果，回退到线性进度映射
        if aligned_template_idx >= 0 and template_window.shape[0] > aligned_template_idx:
            template_lm_for_angle = template_window[aligned_template_idx]
            template_angles = self._angle_calc.compute_frame_angles(
                template_lm_for_angle
            )
        else:
            template_center = self._map_progress(
                frame_index, expected_total_frames
            )
            template_angles = self._get_template_frame_angles(template_center)

        # 4. 角度偏差检测 + 建议生成
        angle_items = self._detect_deviations(
            user_angles, template_angles, now
        )

        # 合并角度和空间建议，角度在前（优先级更高）
        all_items = angle_items + spatial_items

        return FeedbackSnapshot(
            items=all_items,
            has_pose=True,
            window_similarity=window_similarity,
            frame_index=frame_index,
            timestamp=now,
            buffer_size=buffer_size,
            dtw_aligned_template_frame=aligned_template_idx,
            spatial_deviation_overall=spatial_overall,
        )

    # ================================================================
    # 内部方法
    # ================================================================

    def _map_progress(
        self, frame_index: int, expected_total: int
    ) -> int:
        """
        基于时间比例映射到模板帧位置

        Args:
            frame_index: 用户当前帧索引
            expected_total: 预期总帧数

        Returns:
            模板帧索引（含越界保护）
        """
        template_len = self._template_array.shape[0]

        if expected_total <= 0:
            expected_total = template_len

        progress = frame_index / max(expected_total, 1)
        progress = min(progress, 1.0)  # 越界保护

        center = int(progress * (template_len - 1))
        return max(0, min(center, template_len - 1))

    def _get_template_window(
        self, center: int
    ) -> np.ndarray:
        """
        提取模板的固定窗口帧段

        Args:
            center: 中心帧索引

        Returns:
            (W, 33, 4) 模板窗口数组
        """
        template_len = self._template_array.shape[0]
        half = self._window_size // 2

        start = max(0, center - half)
        end = min(template_len, center + half)

        # 确保至少取 buffer 大小的帧
        actual_window = min(len(self._frame_buffer), end - start)
        if actual_window < 1:
            actual_window = 1

        start = max(0, center - actual_window // 2)
        end = min(template_len, start + actual_window)
        start = max(0, end - actual_window)

        return self._template_array[start:end]

    def _get_template_frame_angles(self, frame_index: int) -> Dict[str, float]:
        """获取模板指定帧的关节角度"""
        frame_data = self._template_array[frame_index]  # (33, 4)
        return self._angle_calc.compute_frame_angles(frame_data)

    def _compute_window_dtw(
        self, frame_index: int, expected_total: int
    ) -> Tuple[float, List, np.ndarray, np.ndarray]:
        """
        计算固定窗口 DTW 相似度，同时返回对齐路径和原始窗口数据

        Args:
            frame_index: 当前帧索引
            expected_total: 预期总帧数

        Returns:
            (similarity, path, user_window, template_window)
            - similarity: 相似度 [0, 1]
            - path: DTW 对齐路径 [(i, j), ...]
            - user_window: (W, 33, 4) 用户窗口原始关键点
            - template_window: (W', 33, 4) 模板窗口原始关键点
        """
        # 用户窗口
        user_window = np.stack(list(self._frame_buffer), axis=0)  # (W, 33, 4)

        # 模板窗口
        template_center = self._map_progress(frame_index, expected_total)
        template_window = self._get_template_window(template_center)

        # 确保两个窗口至少有 2 帧
        if user_window.shape[0] < 2 or template_window.shape[0] < 2:
            return 0.0, [], user_window, template_window

        # 转换为特征矩阵 — 使用角度特征（位置无关）
        user_feat = _window_to_angle_matrix(
            user_window, valid_angle_names=self._valid_angle_names
        )
        tmpl_feat = _window_to_angle_matrix(
            template_window, valid_angle_names=self._valid_angle_names
        )

        try:
            distance, path, _ = compute_dtw(
                user_feat, tmpl_feat,
                algorithm=self._algorithm,
                metric="euclidean",
            )

            # 归一化相似度（使用高斯核映射，与离线模式一致）
            path_len = len(path) if path else 1
            norm_dist = distance / path_len
            similarity = float(np.exp(-(norm_dist ** 2) / (2 * 0.7 ** 2)))
            return similarity, path, user_window, template_window

        except Exception as e:
            logger.debug(f"窗口 DTW 计算失败: {e}")
            return 0.0, [], user_window, template_window

    def _normalize_landmark_coords(
        self, lm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单帧关键点坐标归一化：髋中心平移 + 躯干尺度缩放

        Args:
            lm: (33, 4) 单帧关键点 [x, y, z, visibility]

        Returns:
            (coords, vis)
            - coords: (J, 3) 归一化后的核心关节 xyz
            - vis: (J,) 各关节可见度
        """
        J = len(CORE_JOINT_INDICES)
        coords = np.zeros((J, 3), dtype=np.float64)
        vis_arr = np.ones(J, dtype=np.float64)

        if lm.shape[0] <= 24:
            return coords, vis_arr

        # 髋中点
        cx = (lm[23, 0] + lm[24, 0]) / 2.0
        cy = (lm[23, 1] + lm[24, 1]) / 2.0
        cz = (lm[23, 2] + lm[24, 2]) / 2.0

        # 肩中点 → 躯干长度
        sx = (lm[11, 0] + lm[12, 0]) / 2.0
        sy = (lm[11, 1] + lm[12, 1]) / 2.0
        sz = (lm[11, 2] + lm[12, 2]) / 2.0
        torso = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2)
        if torso < 1e-3:
            torso = 1.0

        for k, j in enumerate(CORE_JOINT_INDICES):
            if j < lm.shape[0]:
                coords[k, 0] = (lm[j, 0] - cx) / torso
                coords[k, 1] = (lm[j, 1] - cy) / torso
                coords[k, 2] = (lm[j, 2] - cz) / torso
                vis_arr[k] = float(lm[j, 3])

        return coords, vis_arr

    def _detect_spatial_deviations(
        self,
        user_lm: np.ndarray,
        template_lm: np.ndarray,
        now: float,
    ) -> Tuple[List[FeedbackItem], float]:
        """
        检测空间坐标偏差并生成矫正建议

        使用加权欧氏距离：轴加权 + 关节重要性 + 可见度。

        Args:
            user_lm: (33, 4) 用户当前帧关键点
            template_lm: (33, 4) DTW 对齐的模板帧关键点
            now: 当前时间戳

        Returns:
            (items, overall_deviation)
            - items: 空间矫正建议列表
            - overall_deviation: 整体平均偏差
        """
        u_coords, u_vis = self._normalize_landmark_coords(user_lm)
        t_coords, t_vis = self._normalize_landmark_coords(template_lm)

        J = len(CORE_JOINT_INDICES)
        deviations = np.zeros(J, dtype=np.float64)

        for k in range(J):
            deviations[k] = weighted_euclidean_distance(
                u_coords[k], t_coords[k],
                joint_indices=CORE_JOINT_INDICES,
                k=k,
                visibility=float(u_vis[k]),
            )

        overall = float(np.mean(deviations))

        # 生成建议：取偏差最大的 top-3 关节
        sorted_idx = np.argsort(deviations)[::-1]
        items = []

        for rank, k in enumerate(sorted_idx[:3]):
            dev = float(deviations[k])
            if dev < self._spatial_threshold:
                continue

            joint_idx = CORE_JOINT_INDICES[k]
            en_name = (
                LANDMARK_NAMES[joint_idx]
                if 0 <= joint_idx < len(LANDMARK_NAMES)
                else f"joint_{joint_idx}"
            )
            cn_name = CORE_JOINT_NAMES.get(joint_idx, en_name)

            # 去重检查
            dedup_key = f"spatial_{en_name}"
            last_shown = self._dedup_cache.get(dedup_key, 0.0)
            if (now - last_shown) < self._dedup_interval:
                continue

            # 方向判断（用户关节相对于模板的位置）
            u_xyz = u_coords[k]
            t_xyz = t_coords[k]
            diff = u_xyz - t_xyz
            max_axis = np.argmax(np.abs(diff))
            axis_names = ["X", "Y", "Z"]
            dir_str = (
                f"位置{'偏高' if diff[max_axis] > 0 else '偏低'}"
                f"({axis_names[max_axis]}轴)"
            )

            # 严重等级
            if dev > 0.25:
                severity = Severity.ERROR
            elif dev > 0.15:
                severity = Severity.WARNING
            else:
                severity = Severity.INFO

            items.append(FeedbackItem(
                joint_name=en_name,
                joint_display=cn_name,
                deviation_spatial=dev,
                direction=dir_str,
                advice=f"{cn_name}位置偏差较大，请调整姿态",
                severity=severity,
                source="spatial",
            ))

            self._dedup_cache[dedup_key] = now

        return items, overall

    def _detect_deviations(
        self,
        user_angles: Dict[str, float],
        template_angles: Dict[str, float],
        now: float,
    ) -> List[FeedbackItem]:
        """
        检测角度偏差并生成矫正建议

        Args:
            user_angles: 用户当前帧角度
            template_angles: 模板对应帧角度
            now: 当前时间戳

        Returns:
            去重后的 FeedbackItem 列表
        """
        items = []

        for joint_name in user_angles:
            if joint_name not in template_angles:
                continue

            user_angle = user_angles[joint_name]
            template_angle = template_angles[joint_name]
            deviation = user_angle - template_angle

            # 阈值过滤
            if abs(deviation) < self._angle_threshold:
                continue

            # 将 "left_knee_angle" → "left_knee" 作为 advice_map / 显示名 key
            base_key = joint_name[:-6] if joint_name.endswith("_angle") else joint_name

            # 去重检查
            dedup_key = f"{base_key}_{('small' if deviation < 0 else 'large')}"
            last_shown = self._dedup_cache.get(dedup_key, 0.0)
            if (now - last_shown) < self._dedup_interval:
                continue

            # 生成建议
            direction = "small" if deviation < 0 else "large"
            advice_map = ANGLE_ADVICE.get(base_key, {})
            advice = advice_map.get(
                direction,
                f"{'角度过小' if direction == 'small' else '角度过大'}，请调整"
            )

            # 严重等级
            abs_dev = abs(deviation)
            if abs_dev > 30:
                severity = Severity.ERROR
            elif abs_dev > 20:
                severity = Severity.WARNING
            else:
                severity = Severity.INFO

            items.append(FeedbackItem(
                joint_name=base_key,
                joint_display=JOINT_DISPLAY_NAMES.get(base_key, base_key),
                deviation_deg=abs_dev,
                direction="角度过小" if direction == "small" else "角度过大",
                advice=advice,
                severity=severity,
            ))

            # 更新去重缓存
            self._dedup_cache[dedup_key] = now

        # 按严重等级排序（error > warning > info）
        severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        items.sort(key=lambda x: severity_order.get(x.severity, 3))

        return items
