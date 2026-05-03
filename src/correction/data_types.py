"""
矫正反馈数据类型定义

定义矫正模块的核心数据结构：
- CorrectionItem: 单条矫正建议
- CorrectionReport: 完整矫正报告
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 优先级常量
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

# 优先级排序权重（数值越小越靠前）
PRIORITY_ORDER = {PRIORITY_HIGH: 0, PRIORITY_MEDIUM: 1, PRIORITY_LOW: 2}


@dataclass
class CorrectionItem:
    """
    单条矫正建议

    Attributes:
        joint_name: 关节名称（如 "left_knee"）
        joint_display_name: 关节中文名称（如 "左膝"）
        deviation: 偏差值
        description: 偏差描述（如 "弯曲角度不够"）
        advice: 矫正建议（如 "建议再下蹲约12°"）
        priority: 优先级 "high" / "medium" / "low"
        angle_diff: 角度差（度），可选
    """
    joint_name: str
    joint_display_name: str
    deviation: float
    description: str
    advice: str
    priority: str = PRIORITY_MEDIUM
    angle_diff: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "joint_name": self.joint_name,
            "joint_display_name": self.joint_display_name,
            "deviation": round(self.deviation, 4),
            "description": self.description,
            "advice": self.advice,
            "priority": self.priority,
            "angle_diff": round(self.angle_diff, 1) if self.angle_diff is not None else None,
        }


@dataclass
class CorrectionReport:
    """
    完整矫正报告

    三层结构:
    1. 概要层：动作类别 + 质量评分 + 整体评语
    2. 建议层：按优先级排序的矫正建议列表
    3. 详情层：偏差数据 + 角度数据

    Attributes:
        action_name: 动作类别标识（如 "squat"）
        action_display_name: 动作中文名称（如 "深蹲"）
        quality_score: 质量评分 [0, 100]
        similarity: DTW 相似度 [0, 1]
        overall_comment: 整体评语
        severity: 偏差严重程度 "mild" / "moderate" / "severe"
        corrections: 矫正建议列表（已排序）
        joint_deviations: 各关节偏差 {joint_name: deviation}
        angle_deviations: 各角度偏差 {angle_name: (user_angle, template_angle, diff)}
        confidence: 分类置信度（自动模式下）
    """
    action_name: str = ""
    action_display_name: str = ""
    quality_score: float = 0.0
    similarity: float = 0.0
    overall_comment: str = ""
    severity: str = "mild"
    corrections: List[CorrectionItem] = field(default_factory=list)
    joint_deviations: Dict[str, float] = field(default_factory=dict)
    angle_deviations: Dict[str, tuple] = field(default_factory=dict)
    confidence: float = 0.0

    # 内部数据（供对比视频生成使用，不参与序列化）
    _best_template: object = field(default=None, repr=False)
    _comparison_result: object = field(default=None, repr=False)

    def to_text(self) -> str:
        """
        格式化为可读的中文文本报告

        Returns:
            格式化的报告文本
        """
        lines = [
            "=" * 50,
            "  动作矫正报告",
            "=" * 50,
            "",
            f"【动作类别】{self.action_display_name or self.action_name}",
            f"【质量评分】{self.quality_score:.1f} / 100",
            f"【相似度】  {self.similarity:.1%}",
            f"【偏差程度】{_severity_cn(self.severity)}",
            "",
            f"【整体评语】{self.overall_comment}",
            "",
        ]

        if self.corrections:
            lines.append("-" * 50)
            lines.append("  矫正建议（按优先级排序）")
            lines.append("-" * 50)
            for i, item in enumerate(self.corrections, 1):
                priority_tag = _priority_tag(item.priority)
                lines.append(f"")
                lines.append(f"  {i}. {priority_tag} {item.joint_display_name}")
                lines.append(f"     问题: {item.description}")
                lines.append(f"     建议: {item.advice}")
                if item.angle_diff is not None:
                    lines.append(f"     角度偏差: 约{abs(item.angle_diff):.0f}°")
        else:
            lines.append("  动作表现良好，无需特别矫正。")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        序列化为字典（可 JSON 输出）

        Returns:
            可 json.dumps() 的字典
        """
        return {
            "action_name": self.action_name,
            "action_display_name": self.action_display_name,
            "quality_score": round(self.quality_score, 1),
            "similarity": round(self.similarity, 4),
            "overall_comment": self.overall_comment,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "corrections": [c.to_dict() for c in self.corrections],
            "joint_deviations": {
                k: round(v, 4) for k, v in self.joint_deviations.items()
            },
            "angle_deviations": {
                k: {
                    "user_angle": round(v[0], 1),
                    "template_angle": round(v[1], 1),
                    "diff": round(v[2], 1),
                }
                for k, v in self.angle_deviations.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @property
    def num_corrections(self) -> int:
        return len(self.corrections)

    @property
    def high_priority_count(self) -> int:
        return sum(1 for c in self.corrections if c.priority == PRIORITY_HIGH)


def _severity_cn(severity: str) -> str:
    """偏差严重程度中文"""
    mapping = {"mild": "轻微", "moderate": "中等", "severe": "严重"}
    return mapping.get(severity, severity)


def _priority_tag(priority: str) -> str:
    """优先级标签"""
    mapping = {
        PRIORITY_HIGH: "[❗高]",
        PRIORITY_MEDIUM: "[⚠ 中]",
        PRIORITY_LOW: "[ℹ 低]",
    }
    return mapping.get(priority, f"[{priority}]")