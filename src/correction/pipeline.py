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
        mismatch_similarity_threshold: Optional[float] = None,
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
            mismatch_similarity_threshold: "动作不符合"阈值；None 时从配置读取
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

        # "不符合"判定阈值（从配置读取默认值）
        if mismatch_similarity_threshold is None:
            try:
                from src.utils.config import get_config
                cfg = get_config().get_section("correction")
                mismatch_similarity_threshold = float(
                    cfg.get("mismatch_similarity_threshold", 0.5)
                )
            except Exception:
                mismatch_similarity_threshold = 0.5
        self._mismatch_threshold = float(mismatch_similarity_threshold)

        logger.info(
            f"CorrectionPipeline 初始化完成: algorithm={algorithm}, "
            f"metric={metric}, auto_mode={'可用' if checkpoint_path else '不可用'}, "
            f"mismatch_threshold={self._mismatch_threshold}"
        )

    @property
    def mismatch_threshold(self) -> float:
        return self._mismatch_threshold

    def set_mismatch_threshold(self, value: float):
        self._mismatch_threshold = float(value)

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

        # 存储内部数据供对比视频生成
        report._best_template = best_template
        report._comparison_result = best_result
        report._user_sequence = user_sequence    # 供帧对比查看器使用（原始未预处理）
        report.template_name = best_template_name

        # 把覆盖度相关字段透传到报告，便于 UI / 文本报告展示
        report.raw_similarity = float(getattr(best_result, "raw_similarity", best_result.similarity))
        report.template_coverage = float(getattr(best_result, "template_coverage", 1.0))
        report.coverage_factor = float(getattr(best_result, "coverage_factor", 1.0))

        # ===== 动作"不符合"判定 =====
        # 两种情况触发 mismatch：
        #   a) 完成度过低 → coverage_factor < 0.1（即使差一点点也会被严重惩罚）
        #   b) 姿势质量过低（raw_similarity）→ 动作本身不对
        raw_sim = report.raw_similarity
        too_incomplete = report.coverage_factor < 0.1
        too_dissimilar = raw_sim < self._mismatch_threshold

        if too_incomplete or too_dissimilar:
            report.mismatch = True
            if too_incomplete:
                # 完成度不足优先作为主要原因
                report.mismatch_reason = (
                    f"动作完成度仅 {report.template_coverage:.1%}，"
                    f"低于最低阈值 {self._comparator._coverage_hard_floor:.0%} — "
                    f"用户动作只覆盖了模板的一小段，无法进行有效矫正"
                )
                report.overall_comment = (
                    "本次动作未覆盖模板的足够部分，不建议基于此给出矫正提示。"
                )
            else:
                report.mismatch_reason = (
                    f"相似度 {raw_sim:.2%} 低于不符合阈值 "
                    f"{self._mismatch_threshold:.2%}"
                )
                report.overall_comment = (
                    "本次动作与该模板差异过大，不建议基于此给出矫正提示。"
                )
            # 清空矫正建议
            report.corrections = []

        logger.info(
            f"矫正报告生成完成: 评分={report.quality_score:.1f}, "
            f"相似度={report.similarity:.2%}(raw={raw_sim:.2%}), "
            f"完成度={report.template_coverage:.2%}, "
            f"建议数={report.num_corrections}, 不符合={report.mismatch}"
        )

        return report