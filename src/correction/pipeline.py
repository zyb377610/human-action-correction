"""
端到端矫正分析流水线

串联所有模块：分类 → 模板匹配 → DTW 对比 → 偏差分析 → 角度对比 → 反馈生成。
输入 PoseSequence，输出 CorrectionReport。
"""

import logging
from typing import Optional

from src.pose_estimation.data_types import PoseSequence
from src.data.template_library import TemplateLibrary
from src.action_comparison.comparison import ActionComparator
from src.action_comparison.deviation_analyzer import JointDeviationAnalyzer

from .angle_utils import AngleCalculator
from .data_types import CorrectionReport
from .feedback import FeedbackGenerator
from .rules import CorrectionRuleEngine

logger = logging.getLogger(__name__)


class CorrectionPipeline:
    """
    端到端矫正分析流水线

    两种模式：
    - 自动模式：ActionPredictor 识别类别 → 自动匹配模板 → 分析
    - 指定模式：手动指定动作类别 → 直接匹配模板 → 分析

    使用示例（指定模式）:
        pipeline = CorrectionPipeline(templates_dir="data/templates")
        report = pipeline.analyze(user_sequence, action_name="squat")
        print(report.to_text())

    使用示例（自动模式）:
        pipeline = CorrectionPipeline(
            templates_dir="data/templates",
            checkpoint_path="outputs/checkpoints/best_model.pth",
            model_type="bilstm",
        )
        report = pipeline.analyze(user_sequence)
    """

    def __init__(
        self,
        templates_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_type: str = "bilstm",
        algorithm: str = "dtw",
        metric: str = "euclidean",
        top_k: int = 5,
        preprocess: bool = True,
        target_frames: Optional[int] = 60,
    ):
        """
        Args:
            templates_dir: 模板库根目录
            checkpoint_path: 分类模型 checkpoint 路径（None 则仅支持指定模式）
            model_type: 模型类型（自动模式使用）
            algorithm: DTW 算法
            metric: 距离度量
            top_k: 偏差分析 top-K 关节数
            preprocess: 是否预处理序列
            target_frames: 预处理目标帧数
        """
        # 模板库
        self._library = TemplateLibrary(templates_dir)

        # DTW 对比器
        self._comparator = ActionComparator(
            algorithm=algorithm,
            metric=metric,
            preprocess=preprocess,
            target_frames=target_frames,
        )

        # 偏差分析器
        self._deviation_analyzer = JointDeviationAnalyzer(top_k=top_k)

        # 角度计算器
        self._angle_calculator = AngleCalculator()

        # 规则引擎 + 反馈生成器
        self._rule_engine = CorrectionRuleEngine()
        self._feedback_generator = FeedbackGenerator(self._rule_engine)

        # 分类模型（延迟加载）
        self._predictor = None
        self._checkpoint_path = checkpoint_path
        self._model_type = model_type

        logger.info(
            f"CorrectionPipeline 初始化完成: algorithm={algorithm}, "
            f"metric={metric}, auto_mode={'可用' if checkpoint_path else '不可用'}"
        )

    def _get_predictor(self):
        """延迟加载分类模型"""
        if self._predictor is None:
            if self._checkpoint_path is None:
                raise ValueError(
                    "自动模式需要提供 checkpoint_path。"
                    "请指定 checkpoint_path 或使用 action_name 参数手动指定动作类别。"
                )

            from src.models.predictor import ActionPredictor

            class_names = sorted(self._library.list_actions())
            if not class_names:
                raise ValueError(
                    "模板库为空，无法初始化分类模型。"
                    "请先录入标准动作模板数据。"
                )

            self._predictor = ActionPredictor(
                checkpoint_path=self._checkpoint_path,
                model_type=self._model_type,
                class_names=class_names,
            )

        return self._predictor

    def analyze(
        self,
        user_sequence: PoseSequence,
        action_name: Optional[str] = None,
        progress_callback=None,
    ) -> CorrectionReport:
        """
        执行端到端矫正分析

        Args:
            user_sequence: 用户动作序列
            action_name: 动作类别名称（None 则自动识别）
            progress_callback: 进度回调 fn(step, total, message)，可选

        Returns:
            CorrectionReport

        Raises:
            ValueError: 模板库无数据或指定类别无模板
        """

        def _progress(step: int, msg: str):
            if progress_callback:
                progress_callback(step, 4, msg)

        prediction = None

        # Step 1: 确定动作类别
        _progress(1, "正在分析动作类型…")
        if action_name is None:
            # 自动模式
            predictor = self._get_predictor()
            prediction = predictor.predict(user_sequence)
            action_name = prediction.label
            logger.info(
                f"自动识别: {action_name} "
                f"(置信度: {prediction.confidence:.2%}, "
                f"质量评分: {prediction.quality_score:.1f})"
            )
        else:
            logger.info(f"指定模式: {action_name}")

        # Step 2: 匹配模板
        _progress(2, "正在匹配标准模板…")
        templates = self._library.load_all_templates(action_name)
        if not templates:
            raise ValueError(
                f"动作 '{action_name}' 无可用标准模板，请先录入模板数据。\n"
                f"使用 TemplateLibrary.add_template('{action_name}', sequence, 'standard_01') 添加模板。"
            )

        # Step 3: DTW 对比（取最佳匹配模板）
        _progress(3, "正在DTW对比分析…")
        comparison_results = self._comparator.compare_with_templates(
            user_sequence, self._library, action_name
        )
        best_result = comparison_results[0]  # 相似度最高的
        best_template_name = best_result.template_name
        best_template = templates[best_template_name]

        logger.info(
            f"最佳匹配模板: {best_template_name} "
            f"(相似度: {best_result.similarity:.2%})"
        )

        # Step 4: 偏差分析
        deviation_report = self._deviation_analyzer.analyze(
            user_sequence, best_template, best_result
        )

        # Step 5: 角度对比
        angle_deviations = self._angle_calculator.compare_angles(
            user_sequence, best_template, best_result.path
        )

        # Step 6: 生成反馈报告
        _progress(4, "正在生成矫正报告…")
        report = self._feedback_generator.generate(
            action_name=action_name,
            deviation_report=deviation_report,
            angle_deviations=angle_deviations,
            similarity=best_result.similarity,
            prediction=prediction,
        )

        logger.info(
            f"矫正报告生成完成: 评分={report.quality_score:.1f}, "
            f"建议数={report.num_corrections}"
        )

        return report