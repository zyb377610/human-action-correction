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
    similarity: float                        # 归一化相似度 (0, 1]
    path: List[Tuple[int, int]]              # 最优对齐路径
    cost_matrix: Optional[np.ndarray]        # 累积代价矩阵 (FastDTW 为 None)
    algorithm: str = ""                      # 使用的算法
    metric: str = ""                         # 使用的距离度量
    template_name: str = ""                  # 匹配的模板名称

    # 以下字段在 ActionComparator.compare() 内部填充
    processed_query: Optional[PoseSequence] = None      # 预处理后的用户序列（path 索引对应）
    processed_template: Optional[PoseSequence] = None   # 预处理后的模板序列（path 索引对应）

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
        """
        self._algorithm = algorithm
        self._metric = metric
        self._window_size = window_size
        self._preprocess = preprocess
        self._target_frames = target_frames
        self._similarity_sigma = similarity_sigma

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

        # ======== 改进相似度归一化（高斯核映射） ========
        path_length = len(path)
        normalized_dist = distance / path_length if path_length > 0 else distance

        sigma = self._similarity_sigma
        similarity = np.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))
        similarity = float(np.clip(similarity, 0.0, 1.0))
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
