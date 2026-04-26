"""动作对比模块"""

from src.action_comparison.distance_metrics import (
    euclidean_distance,
    cosine_distance,
    manhattan_distance,
    get_distance_func,
    sequence_to_feature_matrix,
)
from src.action_comparison.dtw_algorithms import (
    classic_dtw,
    fast_dtw,
    derivative_dtw,
    compute_dtw,
)
from src.action_comparison.comparison import (
    ComparisonResult,
    ActionComparator,
)
from src.action_comparison.deviation_analyzer import (
    DeviationReport,
    JointDeviationAnalyzer,
)

__all__ = [
    # 距离度量
    "euclidean_distance",
    "cosine_distance",
    "manhattan_distance",
    "get_distance_func",
    "sequence_to_feature_matrix",
    # DTW 算法
    "classic_dtw",
    "fast_dtw",
    "derivative_dtw",
    "compute_dtw",
    # 对比器
    "ComparisonResult",
    "ActionComparator",
    # 偏差分析
    "DeviationReport",
    "JointDeviationAnalyzer",
]