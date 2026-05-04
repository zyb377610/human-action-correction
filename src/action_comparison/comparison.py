"""
动作对比器

封装完整对比流程：特征转换 → DTW → 相似度归一化 → 结果输出。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose_estimation.data_types import PoseSequence
from src.data.preprocessing import (
    preprocess_pipeline,
    extract_action_segment,
    filter_skeleton_outliers,
)
from src.data.template_library import TemplateLibrary

from .distance_metrics import sequence_to_feature_matrix
from .dtw_algorithms import compute_dtw

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """DTW 对比结果"""

    distance: float                          # DTW 总距离
    similarity: float                        # 归一化相似度 (0, 1]（已应用覆盖度惩罚）
    path: List[Tuple[int, int]]              # 最优对齐路径
    cost_matrix: Optional[np.ndarray]        # 累积代价矩阵 (FastDTW 为 None)
    algorithm: str = ""                      # 使用的算法
    metric: str = ""                         # 使用的距离度量
    template_name: str = ""                  # 匹配的模板名称

    # 以下字段在 ActionComparator.compare() 内部填充
    processed_query: Optional[PoseSequence] = None      # 预处理后的用户序列（path 索引对应）
    processed_template: Optional[PoseSequence] = None   # 预处理后的模板序列（path 索引对应）

    # 原始相似度（未应用覆盖度惩罚）
    raw_similarity: float = 0.0
    # 模板覆盖率 [0, 1]：DTW 匹配路径覆盖的模板帧比例
    template_coverage: float = 1.0
    # 覆盖度惩罚因子 [0, 1]：similarity = raw_similarity * coverage_factor
    coverage_factor: float = 1.0

    @property
    def path_length(self) -> int:
        return len(self.path)

    @property
    def normalized_distance(self) -> float:
        """路径归一化距离"""
        if self.path_length == 0:
            return self.distance
        return self.distance / self.path_length


class ActionComparator:
    """
    动作对比器

    使用示例:
        comparator = ActionComparator(algorithm="dtw", metric="euclidean")
        result = comparator.compare(user_sequence, template_sequence)
        print(f"相似度: {result.similarity:.2%}")
    """

    def __init__(
        self,
        algorithm: str = "dtw",
        metric: str = "euclidean",
        window_size: Optional[int] = None,
        preprocess: bool = True,
        target_frames: Optional[int] = 60,
        similarity_sigma: float = 0.7,
        coverage_min_full: Optional[float] = None,
        coverage_hard_floor: Optional[float] = None,
    ):
        """
        Args:
            algorithm: DTW 算法 — "dtw" / "fastdtw" / "ddtw"
            metric: 距离度量 — "euclidean" / "cosine" / "manhattan"
            window_size: Sakoe-Chiba 带宽约束
            preprocess: 对比前是否预处理序列
            target_frames: 预处理时的归一化帧数 (None 不重采样)
            similarity_sigma: 高斯相似度映射的 sigma 参数，
                             控制容忍度（越大越宽容，推荐 0.5~1.0）
            coverage_min_full: 覆盖度 ≥ 此值时不降分；None 时从配置读取
            coverage_hard_floor: 覆盖度 < 此值时评分归零；None 时从配置读取
        """
        self._algorithm = algorithm
        self._metric = metric
        self._window_size = window_size
        self._preprocess = preprocess
        self._target_frames = target_frames
        self._similarity_sigma = similarity_sigma

        # 从配置读取覆盖度参数默认值
        if coverage_min_full is None or coverage_hard_floor is None:
            try:
                from src.utils.config import get_config
                cfg = get_config().get_section("correction")
                if coverage_min_full is None:
                    coverage_min_full = float(cfg.get("coverage_min_full", 0.70))
                if coverage_hard_floor is None:
                    coverage_hard_floor = float(cfg.get("coverage_hard_floor", 0.30))
            except Exception:
                coverage_min_full = 0.70 if coverage_min_full is None else coverage_min_full
                coverage_hard_floor = 0.30 if coverage_hard_floor is None else coverage_hard_floor
        self._coverage_min_full = float(coverage_min_full)
        self._coverage_hard_floor = float(coverage_hard_floor)

    def compare(
        self,
        query: PoseSequence,
        template: PoseSequence,
        template_name: str = "",
    ) -> ComparisonResult:
        """
        对比两个动作序列

        Args:
            query: 用户动作序列
            template: 标准模板序列
            template_name: 模板名称

        Returns:
            ComparisonResult
        """
        # 记录原始帧数——预处理会压缩，必须按比例保留差异
        orig_q = query.num_frames
        orig_t = template.num_frames

        # ======== 预处理前先过滤骨骼突变帧 + 提取动作片段 ========
        # 1. 过滤多人遮挡导致的骨骼突变帧
        query = filter_skeleton_outliers(query)
        template = filter_skeleton_outliers(template)

        # 2. 自动提取动作片段，剔除无关准备/收尾帧
        query = extract_action_segment(query)
        template = extract_action_segment(template)

        # 记录裁剪后的帧数（用于对比视频的帧映射）
        cropped_q = query.num_frames
        cropped_t = template.num_frames

        logger.debug(
            f"片段提取后: query {orig_q}→{cropped_q}帧, "
            f"template {orig_t}→{cropped_t}帧"
        )
        # ==============================================================

        # 预处理：按原始比例分配目标帧数，确保子序列 DTW 有发挥空间
        if self._preprocess:
            q_target = self._target_frames  # 默认 60
            t_target = self._target_frames
            if cropped_t > 0:
                ratio = cropped_q / cropped_t
                if ratio > 1.2:
                    q_target = int(self._target_frames * ratio)
                elif ratio < 0.8:
                    t_target = int(self._target_frames / ratio)
            query = preprocess_pipeline(query, target_frames=q_target)
            template = preprocess_pipeline(template, target_frames=t_target)

        # 转换为特征矩阵（带身体比例归一化）
        q_matrix = sequence_to_feature_matrix(
            query, normalize_body_scale=True
        )
        t_matrix = sequence_to_feature_matrix(
            template, normalize_body_scale=True
        )

        # DTW 计算（子序列模式：自动双向适配长短不一的序列）
        distance, path, cost_matrix = compute_dtw(
            q_matrix, t_matrix,
            algorithm=self._algorithm,
            metric=self._metric,
            window_size=self._window_size,
            use_subsequence=True,
        )

        # ======== 相似度归一化（高斯核映射） ========
        path_length = len(path)
        normalized_dist = distance / path_length if path_length > 0 else distance

        sigma = self._similarity_sigma
        raw_similarity = np.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))
        raw_similarity = float(np.clip(raw_similarity, 0.0, 1.0))

        # ======== 完成度（基于裁剪后帧数比，预处理前） ========
        # 使用 cropped_q / cropped_t（已剔除前后静止段，但尚未做 target_frames
        # 重采样）作为"真实完成度"。这样能准确区分：
        #   - 用户做完整段动作    → cropped_q ≈ cropped_t → 完成度 ≈ 1.0
        #   - 用户只做了前 30%    → cropped_q ≈ 0.3 * cropped_t → 完成度 ≈ 0.3
        #   - 用户做了很多无关动作+完整动作 → cropped_q > cropped_t → 完成度 = 1.0
        if cropped_t > 0 and path:
            template_coverage = min(cropped_q / float(cropped_t), 1.0)
        else:
            template_coverage = 0.0
        template_coverage = float(np.clip(template_coverage, 0.0, 1.0))

        # ======== 覆盖度函数 f(coverage) — 双阀限 + 二次曲线 ========
        # coverage ≤ hard_floor         → factor = 0（评分归零）
        # coverage ≥ min_full           → factor = 1（不降分）
        # hard_floor < coverage < min_full → 二次曲线插值，接近 min_full 时快速上升
        #   奖励"接近全部完成"的用户
        hard_floor = self._coverage_hard_floor
        min_full = self._coverage_min_full
        if template_coverage <= hard_floor:
            coverage_factor = 0.0
        elif template_coverage >= min_full:
            coverage_factor = 1.0
        else:
            # 归一化到 [0, 1]
            t = (template_coverage - hard_floor) / (min_full - hard_floor)
            # 二次曲线：f(t) = t^2 ∈ [0, 1]，接近 1 时上升快
            # 也可考虑 t*(2-t) 让接近 1 时奖励更大；这里用 t^2 更严格
            coverage_factor = float(t * t)
        coverage_factor = float(np.clip(coverage_factor, 0.0, 1.0))

        # 最终相似度 = 姿势质量 × 完成度函数
        similarity = float(np.clip(raw_similarity * coverage_factor, 0.0, 1.0))

        logger.debug(
            f"DTW: distance={distance:.4f} path_len={path_length} "
            f"raw_sim={raw_similarity:.3f} coverage={template_coverage:.3f} "
            f"factor={coverage_factor:.3f} final_sim={similarity:.3f}"
        )
        # ==================================================

        result = ComparisonResult(
            distance=distance,
            similarity=similarity,
            path=path,
            cost_matrix=cost_matrix,
            algorithm=self._algorithm,
            metric=self._metric,
            template_name=template_name,
            processed_query=query,
            processed_template=template,
            raw_similarity=raw_similarity,
            template_coverage=template_coverage,
            coverage_factor=coverage_factor,
        )

        # 存储裁剪信息，供对比视频生成时使用
        result.query_crop_range = query.metadata.get('cropped_from', (0, orig_q))
        result.template_crop_range = template.metadata.get('cropped_from', (0, orig_t))
        result.cropped_query_frames = cropped_q
        result.cropped_template_frames = cropped_t

        return result

    def compare_with_templates(
        self,
        query: PoseSequence,
        library: TemplateLibrary,
        action: str,
    ) -> List[ComparisonResult]:
        """
        将用户序列与模板库中某动作的所有模板对比

        Args:
            query: 用户动作序列
            library: 模板库
            action: 动作类别名称

        Returns:
            按相似度降序排列的 ComparisonResult 列表
        """
        templates = library.load_all_templates(action)
        if not templates:
            logger.warning(f"动作 '{action}' 无可用模板")
            return []

        results = []
        for name, template_seq in templates.items():
            result = self.compare(query, template_seq, template_name=name)
            results.append(result)
            logger.debug(
                f"对比 {name}: distance={result.distance:.4f}, "
                f"similarity={result.similarity:.2%}"
            )

        # 按相似度降序排列
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results
