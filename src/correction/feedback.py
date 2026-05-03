"""
反馈报告生成器

整合分类结果、DTW 对比、偏差分析和角度对比，
输出用户友好的 CorrectionReport。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.action_comparison.deviation_analyzer import DeviationReport
from src.models.data_types import PredictionResult
from src.utils.config import get_config

from .data_types import (
    CorrectionItem, CorrectionReport,
    PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW, PRIORITY_ORDER,
)
from .rules import CorrectionRuleEngine

logger = logging.getLogger(__name__)

# 动作中文名称映射
ACTION_DISPLAY_NAMES = {
    "squat": "深蹲",
    "arm_raise": "举臂",
    "side_bend": "侧弯",
    "lunge": "弓步",
    "standing_stretch": "站立拉伸",
}

# 质量评分评语模板
QUALITY_COMMENTS = {
    "excellent": "动作完成质量优秀，继续保持！",
    "good": "动作整体良好，有少许细节可以改进。",
    "needs_improvement": "动作存在明显偏差，请根据以下建议进行调整。",
    "poor": "动作偏差较大，建议仔细对照标准动作重新练习。",
}


class FeedbackGenerator:
    """
    反馈报告生成器

    接收各模块的分析结果，生成结构化的 CorrectionReport。

    使用示例:
        generator = FeedbackGenerator()
        report = generator.generate(
            action_name="squat",
            deviation_report=dev_report,
            angle_deviations=angle_devs,
            similarity=0.85,
            prediction=pred_result,
        )
        print(report.to_text())
    """

    def __init__(self, rule_engine: Optional[CorrectionRuleEngine] = None):
        """
        Args:
            rule_engine: 矫正规则引擎实例，None 则创建默认引擎
        """
        self._engine = rule_engine or CorrectionRuleEngine()

        # 从配置文件读取动作显示名称
        config = get_config()
        categories = config.get("actions", {}).get("categories", {})
        self._display_names = dict(ACTION_DISPLAY_NAMES)
        for name, info in categories.items():
            if "display_name" in info:
                self._display_names[name] = info["display_name"]

    def generate(
        self,
        action_name: str,
        deviation_report: DeviationReport,
        angle_deviations: Optional[Dict[str, Tuple[float, float, float]]] = None,
        similarity: float = 0.0,
        prediction: Optional[PredictionResult] = None,
        quality_score: Optional[float] = None,
    ) -> CorrectionReport:
        """
        生成矫正报告

        Args:
            action_name: 动作类别名称
            deviation_report: 逐关节偏差分析报告
            angle_deviations: 角度偏差 {角度名: (用户角度, 模板角度, 差值)}
            similarity: DTW 相似度
            prediction: 分类模型预测结果（自动模式下）
            quality_score: 质量评分（优先使用，不传则从 prediction 或 similarity 推算）

        Returns:
            CorrectionReport
        """
        # 确定质量评分（sigmoid 映射：低相似度低分，中高相似度加速上升）
        if quality_score is not None:
            score = quality_score
        elif prediction is not None:
            score = prediction.quality_score
        else:
            # sigmoid-like 映射 similarity ∈ [0,1] → score ∈ [0,100]
            # sim=0.2→14, 0.4→35, 0.5→50, 0.6→65, 0.7→77, 0.8→86, 0.9→92
            k = 6.0  # 陡峭度
            x0 = 0.5  # 中点
            score = 100.0 / (1.0 + np.exp(-k * (similarity - x0)))

        # 确定置信度
        confidence = prediction.confidence if prediction else 0.0

        # 整体评语
        overall_comment = self._generate_comment(score)

        # 规则匹配生成矫正建议
        corrections = self._engine.match_rules(
            action=action_name,
            joint_deviations=deviation_report.joint_deviations,
            angle_deviations=angle_deviations,
        )

        # 按优先级排序
        corrections = self._sort_corrections(corrections)

        # 动作显示名称
        display_name = self._display_names.get(action_name, action_name)

        return CorrectionReport(
            action_name=action_name,
            action_display_name=display_name,
            quality_score=score,
            similarity=similarity,
            overall_comment=overall_comment,
            severity=deviation_report.severity,
            corrections=corrections,
            joint_deviations=deviation_report.joint_deviations,
            angle_deviations=angle_deviations or {},
            confidence=confidence,
        )

    @staticmethod
    def _generate_comment(score: float) -> str:
        """根据质量评分生成整体评语"""
        if score >= 90:
            return QUALITY_COMMENTS["excellent"]
        elif score >= 70:
            return QUALITY_COMMENTS["good"]
        elif score >= 50:
            return QUALITY_COMMENTS["needs_improvement"]
        else:
            return QUALITY_COMMENTS["poor"]

    @staticmethod
    def _sort_corrections(items: List[CorrectionItem]) -> List[CorrectionItem]:
        """
        按优先级排序，同级内按偏差降序

        Returns:
            排序后的 CorrectionItem 列表
        """
        return sorted(
            items,
            key=lambda c: (PRIORITY_ORDER.get(c.priority, 99), -c.deviation),
        )