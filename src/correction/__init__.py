"""
矫正反馈模块

提供动作矫正的完整能力：
- 规则引擎：基于偏差和角度的规则匹配
- 反馈生成：结构化矫正报告
- 端到端流水线：从 PoseSequence 到 CorrectionReport
- 可视化：偏差骨骼图 + 柱状图
"""

from .data_types import CorrectionItem, CorrectionReport
from .angle_utils import AngleCalculator, compute_angle_3d
from .rules import CorrectionRule, CorrectionRuleEngine
from .feedback import FeedbackGenerator
from .pipeline import CorrectionPipeline
from .report_visualizer import ReportVisualizer

__all__ = [
    # 数据类型
    "CorrectionItem",
    "CorrectionReport",
    # 角度计算
    "AngleCalculator",
    "compute_angle_3d",
    # 规则引擎
    "CorrectionRule",
    "CorrectionRuleEngine",
    # 反馈生成
    "FeedbackGenerator",
    # 流水线
    "CorrectionPipeline",
    # 可视化
    "ReportVisualizer",
]