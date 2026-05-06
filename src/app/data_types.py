"""
应用层数据类型定义

定义 Web 界面与应用流水线之间的数据传递结构：
- AnalysisResult: 完整分析结果
- ProcessedFrame: 单帧处理结果（摄像头模式）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.correction.data_types import CorrectionReport, CorrectionItem


@dataclass
class AnalysisResult:
    """
    完整的端到端分析结果

    Attributes:
        action_name: 动作类别标识（如 "squat"）
        action_display_name: 动作中文名称（如 "深蹲"）
        quality_score: 质量评分 [0, 100]
        similarity: DTW 相似度 [0, 1]
        report_text: 矫正报告文本（中文可读）
        deviation_plot_path: 偏差柱状图文件路径
        skeleton_video_path: 骨骼标注视频文件路径
        corrections: 矫正建议列表
        report: 原始 CorrectionReport 对象
        confidence: 分类置信度（自动模式下）
    """
    action_name: str = ""
    action_display_name: str = ""
    quality_score: float = 0.0
    similarity: float = 0.0
    report_text: str = ""
    deviation_plot_path: Optional[str] = None
    skeleton_video_path: Optional[str] = None
    comparison_video_path: Optional[str] = None
    corrections: List[CorrectionItem] = field(default_factory=list)
    report: Optional[CorrectionReport] = None
    confidence: Optional[float] = None
    algorithm_used: str = ""

    @property
    def num_corrections(self) -> int:
        return len(self.corrections)

    @property
    def has_issues(self) -> bool:
        """是否存在需要矫正的问题"""
        return self.num_corrections > 0

    def summary(self) -> str:
        """简短摘要"""
        lines = [
            f"🏋️ 动作: {self.action_display_name or self.action_name}",
            f"📊 评分: {self.quality_score:.1f} / 100",
        ]
        # 若有 report 对象，展示姿势质量 + 完成度两个维度
        if self.report is not None:
            raw = getattr(self.report, "raw_similarity", None)
            cov = getattr(self.report, "template_coverage", None)
            if raw is not None and cov is not None:
                lines.append(f"   ├─ 姿势质量: {raw:.1%}")
                lines.append(f"   └─ 完成度:  {cov:.1%}")
        lines.append(f"📏 折算相似度: {self.similarity:.1%}")
        lines.append(f"📝 矫正建议: {self.num_corrections} 条")
        if self.algorithm_used:
            lines.append(f"🔧 使用算法: {self.algorithm_used}")
        return "\n".join(lines)


@dataclass
class ProcessedFrame:
    """
    摄像头单帧处理结果

    Attributes:
        annotated_image: 叠加骨骼标注的 BGR 图像
        landmarks: 当前帧的关键点数组 (33, 4)，None 表示未检测到人体
        has_pose: 是否检测到人体姿态
    """
    annotated_image: np.ndarray = field(default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8))
    landmarks: Optional[np.ndarray] = None

    @property
    def has_pose(self) -> bool:
        return self.landmarks is not None


@dataclass
class TemplateRecordResult:
    """
    模板录入结果

    Attributes:
        success: 是否录入成功
        error: 失败时的错误描述
    """
    success: bool = False
    error: str = ""
