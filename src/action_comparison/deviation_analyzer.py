"""
逐关节偏差分析

沿 DTW 对齐路径计算每个关节的偏差，定位偏差最大的关节。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose_estimation.data_types import PoseSequence, LANDMARK_NAMES
from .distance_metrics import sequence_to_feature_matrix, CORE_JOINT_INDICES, CORE_JOINT_NAMES
from .comparison import ComparisonResult

logger = logging.getLogger(__name__)

# 偏差严重程度阈值（基于归一化坐标 + 身体比例归一化后的欧氏距离）
SEVERITY_THRESHOLDS = {
    "mild": 0.08,       # < 0.08: 轻微
    "moderate": 0.18,   # 0.08 ~ 0.18: 中等
    # > 0.18: 严重
}


@dataclass
class DeviationReport:
    """逐关节偏差分析报告"""

    joint_deviations: Dict[str, float]       # 各关节平均偏差 {joint_name: deviation}
    worst_joints: List[str]                  # 偏差最大的 top-K 关节名
    frame_deviations: np.ndarray             # (path_len,) 逐对齐步骤的总偏差曲线
    severity: str                            # 整体偏差严重程度: "mild" / "moderate" / "severe"
    overall_deviation: float                 # 整体平均偏差
    worst_joint_details: List[Dict] = field(default_factory=list)  # top-K 关节详情

    def summary(self) -> str:
        """生成偏差摘要文本"""
        severity_cn = {"mild": "轻微", "moderate": "中等", "severe": "严重"}
        lines = [
            f"偏差程度: {severity_cn.get(self.severity, self.severity)}",
            f"整体平均偏差: {self.overall_deviation:.4f}",
            f"偏差最大关节:",
        ]
        for detail in self.worst_joint_details:
            lines.append(
                f"  - {detail['name']} (#{detail['index']}): "
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

    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k: 输出偏差最大的前 K 个关节
        """
        self._top_k = top_k

    def analyze(
        self,
        query: PoseSequence,
        template: PoseSequence,
        result: ComparisonResult,
    ) -> DeviationReport:
        """
        沿对齐路径分析逐关节偏差

        Args:
            query: 用户动作序列（已预处理）
            template: 标准模板序列（已预处理）
            result: DTW 对比结果（含对齐路径）

        Returns:
            DeviationReport
        """
        q_matrix = sequence_to_feature_matrix(query)   # (N, 36) 12 核心关节
        t_matrix = sequence_to_feature_matrix(template) # (M, 36)
        path = result.path

        num_joints = len(CORE_JOINT_INDICES)
        joint_diffs = np.zeros((len(path), num_joints), dtype=np.float64)

        # 沿对齐路径，对每一步计算每个核心关节的欧氏距离
        for step, (i, j) in enumerate(path):
            for k in range(num_joints):
                q_xyz = q_matrix[i, k * 3: k * 3 + 3]
                t_xyz = t_matrix[j, k * 3: k * 3 + 3]
                joint_diffs[step, k] = float(np.sqrt(np.sum((q_xyz - t_xyz) ** 2)))

        # 每个关节的平均偏差
        joint_mean = np.mean(joint_diffs, axis=0)  # (12,)

        # 构建 {关节名: 偏差} 字典
        joint_deviations = {}
        for k in range(num_joints):
            joint_idx = CORE_JOINT_INDICES[k]
            name = CORE_JOINT_NAMES.get(
                joint_idx,
                LANDMARK_NAMES[joint_idx] if joint_idx < len(LANDMARK_NAMES) else f"joint_{joint_idx}"
            )
            joint_deviations[name] = float(joint_mean[k])

        # 排序取 top-K
        sorted_joints = sorted(
            joint_deviations.items(), key=lambda x: x[1], reverse=True
        )
        worst_joints = [name for name, _ in sorted_joints[: self._top_k]]

        # top-K 详细信息
        worst_joint_details = []
        for name, dev in sorted_joints[: self._top_k]:
            # 反查索引
            idx = next(
                (k for k, n in enumerate(LANDMARK_NAMES) if n == name), -1
            )
            worst_joint_details.append({
                "name": name,
                "index": idx,
                "deviation": dev,
            })

        # 逐步总偏差曲线
        frame_deviations = np.sum(joint_diffs, axis=1)  # (path_len,)

        # 整体平均偏差
        overall_deviation = float(np.mean(joint_mean))

        # 严重程度分级
        severity = self._classify_severity(overall_deviation)

        return DeviationReport(
            joint_deviations=joint_deviations,
            worst_joints=worst_joints,
            frame_deviations=frame_deviations,
            severity=severity,
            overall_deviation=overall_deviation,
            worst_joint_details=worst_joint_details,
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
