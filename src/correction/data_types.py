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

    # 关节中文名映射（供格式化显示用）
    joint_display_map: Dict[str, str] = field(default_factory=dict)

    # 逐帧对齐详情（FrameDeviationDetail 列表），便于按帧展开建议
    frame_details: List = field(default_factory=list)

    # 模板权威性：被模板"声明"为不可见、因而未参与对比/建议的关节中文名
    excluded_joints: List[str] = field(default_factory=list)

    # 模板信息（可选，用于报告展示）
    template_name: str = ""

    # 问题5 新增：若判定"动作不符合模板"，则不再给出具体矫正建议
    mismatch: bool = False
    mismatch_reason: str = ""

    # === 完成度相关（来自 ComparisonResult）===
    raw_similarity: float = 0.0        # 未应用覆盖度惩罚前的相似度
    template_coverage: float = 1.0     # 用户动作覆盖模板的比例 [0, 1]
    coverage_factor: float = 1.0       # 覆盖度惩罚因子 [0, 1]

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
            "=" * 60,
            "  动作矫正报告",
            "=" * 60,
            "",
            f"【动作类别】{self.action_display_name or self.action_name}",
            f"【质量评分】{self.quality_score:.1f} / 100",
            "",
            "【评分维度】",
            f"  ├─ 姿势质量 {self.raw_similarity:.1%}  "
            f"（仅看动作做得对不对，忽略完成度）",
            f"  ├─ 完成度   {self.template_coverage:.1%}  "
            f"（用户动作覆盖了模板的比例）",
            f"  └─ 折算后相似度 {self.similarity:.1%}  "
            f"（= 姿势质量 × 完成度函数 {self.coverage_factor:.0%}）"
            if self.coverage_factor < 0.999 else
            f"  └─ 折算后相似度 {self.similarity:.1%}  （完成度达标，无折算）",
            "",
            f"【偏差程度】{_severity_cn(self.severity)}",
        ]
        if self.template_name:
            lines.append(f"【匹配模板】{self.template_name}")
        if self.excluded_joints:
            lines.append(
                "【模板关注范围】已自动剔除模板中不可见的关节：" +
                "、".join(self.excluded_joints)
            )
        lines.extend([
            "",
            f"【整体评语】{self.overall_comment}",
            "",
        ])

        # ========== 动作不符合模式：不给出具体建议 ==========
        if self.mismatch:
            lines.append("-" * 60)
            lines.append("  ⛔ 动作不符合")
            lines.append("-" * 60)
            lines.append(
                f"  与「{self.action_display_name or self.action_name}」"
                "模板的相似度过低，系统认为本次动作与该模板不是同一类动作，"
                "因此不再给出具体矫正建议，避免误导。"
            )
            if self.mismatch_reason:
                lines.append(f"  判定依据: {self.mismatch_reason}")
            lines.append("")
            lines.append("  👉 建议：请重新确认所选动作类型，或换个与目标"
                         "模板更接近的动作重试。")
            lines.append("")
            lines.append("=" * 60)
            return "\n".join(lines)

        # ========== 矫正建议 ==========
        if self.corrections:
            lines.append("-" * 60)
            lines.append(f"  矫正建议（共 {len(self.corrections)} 条，按优先级排序）")
            lines.append("-" * 60)
            for i, item in enumerate(self.corrections, 1):
                priority_tag = _priority_tag(item.priority)
                lines.append("")
                lines.append(
                    f"  {i}. {priority_tag} {item.joint_display_name} "
                    f"(偏差 {item.deviation:.4f})"
                )
                lines.append(f"     问题: {item.description}")
                lines.append(f"     建议: {item.advice}")
                if item.angle_diff is not None:
                    lines.append(f"     角度偏差: 约 {abs(item.angle_diff):.0f}°")
        else:
            lines.append("  ✅ 动作表现良好，无需特别矫正。")

        # ========== 逐关节偏差明细 ==========
        if self.joint_deviations:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  逐关节偏差明细（按偏差降序）")
            lines.append("-" * 60)
            sorted_joints = sorted(
                self.joint_deviations.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for en_name, dev in sorted_joints:
                cn = self.joint_display_map.get(en_name, en_name)
                level = _deviation_level(dev)
                lines.append(
                    f"  {level}  {cn:<6s} ({en_name:<20s}) "
                    f"偏差={dev:.4f}"
                )

        # ========== 关节角度对比 ==========
        if self.angle_deviations:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  关节角度对比（用户 vs 模板）")
            lines.append("-" * 60)
            from src.correction.angle_utils import ANGLE_DISPLAY_NAMES
            for name, vals in self.angle_deviations.items():
                if not (isinstance(vals, (tuple, list)) and len(vals) == 3):
                    continue
                u, t, d = vals
                cn = ANGLE_DISPLAY_NAMES.get(name, name)
                tag = "⚠" if abs(d) >= 10 else " "
                lines.append(
                    f"  {tag} {cn:<8s} 用户 {u:6.1f}° | 模板 {t:6.1f}° | 差 {d:+6.1f}°"
                )

        # ========== 逐帧对齐详情 ==========
        if self.frame_details:
            lines.append("")
            lines.append("-" * 60)
            lines.append(
                f"  逐帧对齐明细（共 {len(self.frame_details)} 步，"
                f"仅列出偏差最大的关节）"
            )
            lines.append("-" * 60)
            lines.append(
                f"  {'步':>4} | {'用户帧':>6} | {'模板帧':>6} | "
                f"{'偏差最大关节':<10s} | {'该关节偏差':>10s} | {'总偏差':>8s}"
            )
            for step, d in enumerate(self.frame_details, 1):
                mark = _deviation_level(d.worst_joint_deviation)
                lines.append(
                    f"  {step:>4d} | {d.user_frame:>6d} | {d.template_frame:>6d} | "
                    f"{mark} {d.worst_joint_display:<8s} | "
                    f"{d.worst_joint_deviation:>10.4f} | {d.total_deviation:>8.4f}"
                )

        lines.append("")
        lines.append("=" * 60)
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


def _deviation_level(dev: float) -> str:
    """根据偏差值返回一个视觉标签，用于文本报告排版"""
    if dev >= 0.18:
        return "🔴"
    if dev >= 0.08:
        return "🟡"
    return "🟢"