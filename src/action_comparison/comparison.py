"""
动作对比器

封装完整对比流程：特征转换 → DTW → 相似度归一化 → 结果输出。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose_estimation.data_types import PoseSequence
from src.data.preprocessing import preprocess_pipeline
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
    ):
        """
        Args:
            algorithm: DTW 算法 — "dtw" / "fastdtw" / "ddtw"
            metric: 距离度量 — "euclidean" / "cosine" / "manhattan"
            window_size: Sakoe-Chiba 带宽约束
            preprocess: 对比前是否预处理序列
            target_frames: 预处理时的归一化帧数 (None 不重采样)
        """
        self._algorithm = algorithm
        self._metric = metric
        self._window_size = window_size
        self._preprocess = preprocess
        self._target_frames = target_frames

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
        # 预处理
        if self._preprocess:
            query = preprocess_pipeline(query, target_frames=self._target_frames)
            template = preprocess_pipeline(template, target_frames=self._target_frames)

        # 转换为特征矩阵（带身体比例归一化）
        q_matrix = sequence_to_feature_matrix(
            query, normalize_body_scale=True
        )
        t_matrix = sequence_to_feature_matrix(
            template, normalize_body_scale=True
        )

        # DTW 计算（子序列模式：query 长于 template 时自动找最佳匹配段）
        distance, path, cost_matrix = compute_dtw(
            q_matrix, t_matrix,
            algorithm=self._algorithm,
            metric=self._metric,
            window_size=self._window_size,
            use_subsequence=True,
        )

        # 相似度归一化
        path_length = len(path)
        normalized_dist = distance / path_length if path_length > 0 else distance
        similarity = 1.0 / (1.0 + normalized_dist)

        return ComparisonResult(
            distance=distance,
            similarity=similarity,
            path=path,
            cost_matrix=cost_matrix,
            algorithm=self._algorithm,
            metric=self._metric,
            template_name=template_name,
        )

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
