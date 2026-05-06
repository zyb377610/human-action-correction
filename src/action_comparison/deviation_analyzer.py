"""
逐关节偏差分析

沿 DTW 对齐路径计算每个关节的空间偏差，定位偏差最大的关节。

关键修复：
- 使用 sequence_to_landmark_matrix 得到 (T, J*3) 坐标矩阵（带身体尺度归一化），
  而不是把角度矩阵误当作坐标矩阵（这曾导致只能报出鼻子/左肩/右肩）
- joint_deviations 的 key 使用英文标识（left_shoulder 等），
  使规则引擎能正确匹配
- 额外输出逐帧对齐细节，供矫正报告按帧展开建议
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose_estimation.data_types import PoseSequence, LANDMARK_NAMES
from .distance_metrics import (
    sequence_to_landmark_matrix,
    sequence_to_landmark_matrix_masked,
    sequence_to_landmark_matrix_weighted,
    weighted_euclidean_distance,
    compute_valid_joints,
    CORE_JOINT_INDICES,
    CORE_JOINT_NAMES,
)
from .comparison import ComparisonResult

logger = logging.getLogger(__name__)

# 偏差严重程度阈值（归一化坐标系下的欧氏距离）
SEVERITY_THRESHOLDS = {
    "mild": 0.08,       # < 0.08: 轻微
    "moderate": 0.18,   # 0.08 ~ 0.18: 中等
    # > 0.18: 严重
}


@dataclass
class FrameDeviationDetail:
    """单步对齐的偏差详情"""
    user_frame: int                    # 用户序列帧号
    template_frame: int                # 模板序列帧号
    total_deviation: float             # 该步总偏差（所有核心关节偏差之和）
    worst_joint: str = ""              # 偏差最大的关节英文名
    worst_joint_display: str = ""      # 中文名
    worst_joint_deviation: float = 0.0
    severity: str = "mild"             # 该步严重度：mild/moderate/severe


@dataclass
class DeviationReport:
    """逐关节偏差分析报告"""

    joint_deviations: Dict[str, float]       # 各关节平均偏差 {joint_en_name: deviation}
    worst_joints: List[str]                  # 偏差最大的 top-K 关节名
    frame_deviations: np.ndarray             # (path_len,) 逐对齐步骤的总偏差曲线
    severity: str                            # 整体偏差严重程度: "mild" / "moderate" / "severe"
    overall_deviation: float                 # 整体平均偏差
    worst_joint_details: List[Dict] = field(default_factory=list)   # top-K 关节详情
    frame_details: List[FrameDeviationDetail] = field(default_factory=list)  # 每步对齐详情
    valid_joint_indices: List[int] = field(default_factory=list)    # 模板确定的有效关节索引
    excluded_joint_names: List[str] = field(default_factory=list)   # 因模板不可见而被剔除的关节中文名
    temporal_volatility: float = 0.0         # 时域波动率（帧间偏差变化率的均值），0 表示完全稳定
    use_weighted: bool = False               # 是否使用了加权距离

    def summary(self) -> str:
        """生成偏差摘要文本"""
        severity_cn = {"mild": "轻微", "moderate": "中等", "severe": "严重"}
        lines = [
            f"偏差程度: {severity_cn.get(self.severity, self.severity)}",
            f"整体平均偏差: {self.overall_deviation:.4f}",
        ]
        if self.temporal_volatility > 0:
            stability = "稳定" if self.temporal_volatility < 0.05 else (
                "略有波动" if self.temporal_volatility < 0.12 else "波动较大"
            )
            lines.append(f"动作稳定性: {stability} (波动率={self.temporal_volatility:.4f})")
        lines.append(f"偏差最大关节:")
        for detail in self.worst_joint_details:
            lines.append(
                f"  - {detail['display_name']} ({detail['name']}): "
                f"偏差 {detail['deviation']:.4f}"
            )
        return "\n".join(lines)


class JointDeviationAnalyzer:
    """
    逐关节偏差分析器

    使用示例:
        analyzer = JointDeviationAnalyzer(top_k=5)
        report = analyzer.analyze(query_seq, template_seq, comparison_result)
        print(report.summary())
    """

    def __init__(self, top_k: int = 5, use_weighted: bool = True):
        """
        Args:
            top_k: 输出偏差最大的前 K 个关节
            use_weighted: 是否使用加权欧氏距离（轴加权 + 关节重要性 + 可见度）
        """
        self._top_k = top_k
        self._use_weighted = use_weighted

    def analyze(
        self,
        query: PoseSequence,
        template: PoseSequence,
        result: ComparisonResult,
    ) -> DeviationReport:
        """
        沿对齐路径分析逐关节偏差

        Args:
            query: 用户动作序列（外层传入的原始序列，仅作 fallback）
            template: 标准模板序列（外层传入的原始序列，仅作 fallback）
            result: DTW 对比结果（含对齐路径、预处理后的序列）

        Returns:
            DeviationReport
        """
        # 优先使用 result 中保存的预处理后序列，因为 result.path 的索引是
        # 针对预处理序列的，若直接使用未预处理的 query/template 会出现
        # 帧对齐错位（历史遗留 bug）。
        q_seq = getattr(result, "processed_query", None) or query
        t_seq = getattr(result, "processed_template", None) or template

        # === 模板权威性原则 ===
        # 以模板视频的可见关节为准，自动剔除模板中长期不可见的关节
        # （例如只录了上半身/下半身的模板）。
        valid_joints = compute_valid_joints(t_seq)
        num_joints = len(valid_joints)
        excluded = [
            CORE_JOINT_NAMES.get(j, str(j))
            for j in CORE_JOINT_INDICES if j not in valid_joints
        ]

        # 使用增强版坐标矩阵：髋中心归一化 + 躯干尺度归一化 + 肩轴 2D 旋转对齐
        if self._use_weighted:
            # 加权模式：同时提取坐标和可见度矩阵
            q_matrix, q_vis = sequence_to_landmark_matrix_weighted(
                q_seq, valid_joints, align_shoulder=True
            )  # (N, J*3), (N, J)
            t_matrix, t_vis = sequence_to_landmark_matrix_weighted(
                t_seq, valid_joints, align_shoulder=True
            )  # (M, J*3), (M, J)
        else:
            # 原始模式（向后兼容）
            q_matrix = sequence_to_landmark_matrix_masked(
                q_seq, valid_joints, align_shoulder=True
            )  # (N, J*3)
            t_matrix = sequence_to_landmark_matrix_masked(
                t_seq, valid_joints, align_shoulder=True
            )  # (M, J*3)
            q_vis = None
            t_vis = None

        path = result.path

        joint_indices = list(valid_joints)

        if len(path) == 0:
            # 退化为逐帧对齐（min(N,M)）
            L = min(q_matrix.shape[0], t_matrix.shape[0])
            path = [(i, i) for i in range(L)]

        joint_diffs = np.zeros((len(path), num_joints), dtype=np.float64)

        # 沿对齐路径，对每一步计算每个核心关节的偏差
        for step, (i, j) in enumerate(path):
            # 防御越界
            i = min(i, q_matrix.shape[0] - 1)
            j = min(j, t_matrix.shape[0] - 1)
            for k in range(num_joints):
                q_xyz = q_matrix[i, k * 3: k * 3 + 3]
                t_xyz = t_matrix[j, k * 3: k * 3 + 3]

                if self._use_weighted:
                    # 加权欧氏距离：轴加权 + 关节重要性 + 可见度
                    vis = float(q_vis[i, k]) if q_vis is not None else 1.0
                    joint_diffs[step, k] = weighted_euclidean_distance(
                        q_xyz, t_xyz,
                        joint_indices=joint_indices,
                        k=k,
                        visibility=vis,
                    )
                else:
                    joint_diffs[step, k] = float(
                        np.sqrt(np.sum((q_xyz - t_xyz) ** 2))
                    )

        # 每个关节的平均偏差
        joint_mean = np.mean(joint_diffs, axis=0)  # (num_joints,)

        # 构建 {关节英文名: 偏差} 字典（使用 LANDMARK_NAMES 的英文标识）
        joint_deviations: Dict[str, float] = {}
        english_names: List[str] = []
        display_names: List[str] = []
        for k in range(num_joints):
            joint_idx = joint_indices[k]
            en_name = (
                LANDMARK_NAMES[joint_idx]
                if 0 <= joint_idx < len(LANDMARK_NAMES)
                else f"joint_{joint_idx}"
            )
            cn_name = CORE_JOINT_NAMES.get(joint_idx, en_name)
            joint_deviations[en_name] = float(joint_mean[k])
            english_names.append(en_name)
            display_names.append(cn_name)

        # 排序取 top-K
        sorted_items = sorted(
            joint_deviations.items(), key=lambda x: x[1], reverse=True
        )
        worst_joints = [name for name, _ in sorted_items[: self._top_k]]

        # top-K 详细信息
        worst_joint_details: List[Dict] = []
        en_to_idx = {en: i for i, en in enumerate(english_names)}
        for name, dev in sorted_items[: self._top_k]:
            k = en_to_idx.get(name, -1)
            joint_idx = joint_indices[k] if k >= 0 else -1
            worst_joint_details.append({
                "name": name,
                "display_name": display_names[k] if k >= 0 else name,
                "index": joint_idx,
                "deviation": dev,
            })

        # 逐步总偏差曲线
        frame_deviations = np.sum(joint_diffs, axis=1)  # (path_len,)

        # 整体平均偏差
        overall_deviation = float(np.mean(joint_mean))

        # === 时域波动率 ===
        # 沿路径偏差的一阶差分均值，衡量动作稳定性
        temporal_volatility = 0.0
        if len(frame_deviations) >= 2:
            temporal_volatility = float(
                np.mean(np.abs(np.diff(frame_deviations)))
            )

        # 严重程度分级
        severity = self._classify_severity(overall_deviation)

        # ===== 逐步对齐详情 =====
        frame_details: List[FrameDeviationDetail] = []
        for step, (i, j) in enumerate(path):
            row = joint_diffs[step]
            worst_k = int(np.argmax(row))
            step_total = float(np.sum(row))
            worst_dev = float(row[worst_k])
            frame_details.append(FrameDeviationDetail(
                user_frame=int(i),
                template_frame=int(j),
                total_deviation=step_total,
                worst_joint=english_names[worst_k],
                worst_joint_display=display_names[worst_k],
                worst_joint_deviation=worst_dev,
                severity=self._classify_severity(worst_dev),
            ))

        return DeviationReport(
            joint_deviations=joint_deviations,
            worst_joints=worst_joints,
            frame_deviations=frame_deviations,
            severity=severity,
            overall_deviation=overall_deviation,
            worst_joint_details=worst_joint_details,
            frame_details=frame_details,
            valid_joint_indices=list(joint_indices),
            excluded_joint_names=excluded,
            temporal_volatility=temporal_volatility,
            use_weighted=self._use_weighted,
        )

    @staticmethod
    def _classify_severity(overall_deviation: float) -> str:
        """偏差严重程度分级"""
        if overall_deviation < SEVERITY_THRESHOLDS["mild"]:
            return "mild"
        elif overall_deviation < SEVERITY_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "severe"
