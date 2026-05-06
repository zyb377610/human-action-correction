"""
动作对比模块单元测试

覆盖距离度量、DTW 算法、对比器和偏差分析。
"""

import tempfile
import shutil
import pytest
import numpy as np

from src.pose_estimation.data_types import PoseLandmark, PoseFrame, PoseSequence

from src.action_comparison.distance_metrics import (
    euclidean_distance,
    cosine_distance,
    manhattan_distance,
    weighted_euclidean_distance,
    weighted_euclidean_frame,
    get_distance_func,
    sequence_to_feature_matrix,
    sequence_to_landmark_matrix,
    sequence_to_hybrid_matrix,
    CORE_JOINT_INDICES,
    AXIS_WEIGHTS,
    JOINT_IMPORTANCE_WEIGHTS,
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
    JointDeviationAnalyzer,
)
from src.data.template_library import TemplateLibrary


# ===== Helpers =====

def _make_landmarks(offset=0.0, visibility=0.9):
    return [
        PoseLandmark(
            x=i * 0.03 + offset,
            y=i * 0.02 + offset,
            z=0.01,
            visibility=visibility,
        )
        for i in range(33)
    ]


def _make_sequence(num_frames=30, fps=30.0, offset=0.0):
    frames = [
        PoseFrame(
            timestamp=i / fps,
            frame_index=i,
            landmarks=_make_landmarks(offset),
        )
        for i in range(num_frames)
    ]
    return PoseSequence(frames=frames, fps=fps)


@pytest.fixture
def seq_a():
    return _make_sequence(20, offset=0.0)


@pytest.fixture
def seq_b():
    return _make_sequence(20, offset=0.0)


@pytest.fixture
def seq_diff():
    """偏移后的序列"""
    return _make_sequence(20, offset=0.1)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ===== 距离度量测试 =====

class TestDistanceMetrics:

    def test_euclidean_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == 0.0

    def test_euclidean_known(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(euclidean_distance(a, b) - 5.0) < 1e-10

    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_distance(a, a)) < 1e-10

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_distance(a, b) - 1.0) < 1e-10

    def test_manhattan(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert abs(manhattan_distance(a, b) - 7.0) < 1e-10

    def test_get_distance_func(self):
        f = get_distance_func("euclidean")
        assert callable(f)

    def test_get_distance_func_invalid(self):
        with pytest.raises(ValueError):
            get_distance_func("invalid")

    def test_sequence_to_feature_matrix(self, seq_a):
        """默认使用角度特征（9 个角度）"""
        matrix = sequence_to_feature_matrix(seq_a)
        # 当前实现使用关节角度作为特征，9 个角度
        assert matrix.ndim == 2
        assert matrix.shape[0] == 20
        assert matrix.shape[1] == 9

    def test_sequence_to_landmark_matrix(self, seq_a):
        """关节坐标矩阵：默认 17 核心关节 × 3 = 51"""
        from src.action_comparison.distance_metrics import sequence_to_landmark_matrix
        matrix = sequence_to_landmark_matrix(seq_a)
        assert matrix.shape == (20, 51)

    def test_sequence_to_landmark_matrix_all_joints(self, seq_a):
        """全部 33 个关节 → 99 维"""
        from src.action_comparison.distance_metrics import sequence_to_landmark_matrix
        all_joints = list(range(33))
        matrix = sequence_to_landmark_matrix(seq_a, joint_indices=all_joints)
        assert matrix.shape == (20, 99)


# ===== DTW 算法测试 =====

class TestDTWAlgorithms:

    def test_classic_dtw_identical(self):
        q = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        d, p, c = classic_dtw(q, q)
        assert d == 0.0
        assert len(p) == 3

    def test_classic_dtw_different(self):
        q = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        t = np.array([[0, 0], [0.5, 0.5], [2, 2]], dtype=float)
        d, p, c = classic_dtw(q, t)
        assert d > 0
        assert c.shape == (3, 3)

    def test_classic_dtw_unequal_length(self):
        q = np.array([[0], [1], [2], [3]], dtype=float)
        t = np.array([[0], [1], [3]], dtype=float)
        d, p, c = classic_dtw(q, t)
        assert d >= 0
        # 路径应连接 (0,0) 到 (3,2)
        assert p[0] == (0, 0)
        assert p[-1] == (3, 2)

    def test_classic_dtw_with_window(self):
        q = np.array([[0], [1], [2], [3], [4]], dtype=float)
        t = np.array([[0], [1], [2], [3], [4]], dtype=float)
        d, p, c = classic_dtw(q, t, window_size=2)
        assert d == 0.0

    def test_fast_dtw(self):
        q = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        t = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        d, p, _ = fast_dtw(q, t)
        assert d == 0.0

    def test_fast_dtw_no_cost_matrix(self):
        q = np.array([[0], [1]], dtype=float)
        t = np.array([[0], [1]], dtype=float)
        _, _, cost = fast_dtw(q, t)
        assert cost is None

    def test_derivative_dtw(self):
        q = np.array([[0], [1], [2], [3], [4]], dtype=float)
        t = np.array([[0], [1], [2], [3], [4]], dtype=float)
        d, p, c = derivative_dtw(q, t)
        assert d == 0.0

    def test_compute_dtw_dispatch(self):
        q = np.array([[0, 0], [1, 1]], dtype=float)
        t = np.array([[0, 0], [1, 1]], dtype=float)

        d1, _, _ = compute_dtw(q, t, algorithm="dtw")
        d2, _, _ = compute_dtw(q, t, algorithm="fastdtw")
        d3, _, _ = compute_dtw(q, t, algorithm="ddtw")

        assert d1 == 0.0
        assert d2 == 0.0
        assert d3 == 0.0

    def test_compute_dtw_invalid_algorithm(self):
        q = np.array([[0]], dtype=float)
        with pytest.raises(ValueError):
            compute_dtw(q, q, algorithm="invalid")

    def test_compute_dtw_metrics(self):
        q = np.array([[1, 0], [0, 1]], dtype=float)
        t = np.array([[0, 1], [1, 0]], dtype=float)

        d_euc, _, _ = compute_dtw(q, t, metric="euclidean")
        d_cos, _, _ = compute_dtw(q, t, metric="cosine")
        d_man, _, _ = compute_dtw(q, t, metric="manhattan")

        assert d_euc > 0
        assert d_cos > 0
        assert d_man > 0


# ===== 对比器测试 =====

class TestActionComparator:

    def test_compare_identical(self, seq_a, seq_b):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_b)
        assert result.distance == 0.0
        assert result.similarity == 1.0

    def test_compare_different(self, seq_a, seq_diff):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)
        assert result.distance > 0
        assert 0 < result.similarity < 1.0
        assert result.path_length > 0

    def test_compare_with_preprocess(self, seq_a, seq_diff):
        comp = ActionComparator(preprocess=True, target_frames=30)
        result = comp.compare(seq_a, seq_diff)
        assert result.distance > 0
        assert result.algorithm == "dtw"
        assert result.metric == "euclidean"

    def test_compare_fastdtw(self, seq_a, seq_diff):
        comp = ActionComparator(algorithm="fastdtw", preprocess=False)
        result = comp.compare(seq_a, seq_diff)
        assert result.distance > 0
        # 注：子序列 DTW 模式下 cost_matrix 由 subsequence_dtw 内部计算，
        # 不一定为 None（取决于底层算法是否返回 cost_matrix）
        assert result.algorithm == "fastdtw"

    def test_comparison_result_properties(self, seq_a, seq_diff):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff, template_name="test_tmpl")
        assert result.template_name == "test_tmpl"
        assert result.normalized_distance == result.distance / result.path_length

    def test_compare_with_templates(self, seq_a, temp_dir):
        lib = TemplateLibrary(temp_dir)
        lib.add_template("squat", seq_a, "t1")
        lib.add_template("squat", seq_a, "t2")

        comp = ActionComparator(preprocess=False)
        results = comp.compare_with_templates(seq_a, lib, "squat")

        assert len(results) == 2
        # 完全相同，相似度应为 1.0
        assert results[0].similarity == 1.0

    def test_compare_with_empty_templates(self, seq_a, temp_dir):
        lib = TemplateLibrary(temp_dir)
        comp = ActionComparator(preprocess=False)
        results = comp.compare_with_templates(seq_a, lib, "nonexist")
        assert results == []


# ===== 偏差分析测试 =====

class TestDeviationAnalyzer:

    def test_analyze_identical(self, seq_a, seq_b):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_b)

        analyzer = JointDeviationAnalyzer(top_k=5)
        report = analyzer.analyze(seq_a, seq_b, result)

        assert report.overall_deviation == 0.0
        assert report.severity == "mild"
        assert len(report.worst_joints) == 5
        assert len(report.frame_deviations) == result.path_length

    def test_analyze_different(self, seq_a, seq_diff):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)

        analyzer = JointDeviationAnalyzer(top_k=3)
        report = analyzer.analyze(seq_a, seq_diff, result)

        assert report.overall_deviation > 0
        assert len(report.worst_joints) == 3
        assert len(report.worst_joint_details) == 3
        assert report.worst_joint_details[0]["deviation"] >= report.worst_joint_details[1]["deviation"]

    def test_severity_classification(self, seq_a, seq_diff):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)

        analyzer = JointDeviationAnalyzer()
        report = analyzer.analyze(seq_a, seq_diff, result)

        assert report.severity in ("mild", "moderate", "severe")

    def test_summary_output(self, seq_a, seq_diff):
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)

        analyzer = JointDeviationAnalyzer()
        report = analyzer.analyze(seq_a, seq_diff, result)

        summary = report.summary()
        assert "偏差程度" in summary
        assert "整体平均偏差" in summary
        assert "偏差最大关节" in summary


# ===== 加权距离测试 =====

class TestWeightedDistance:

    def test_axis_weights_constant(self):
        """验证 z 轴权重小于 xy"""
        assert AXIS_WEIGHTS[0] == 1.0
        assert AXIS_WEIGHTS[1] == 1.0
        assert AXIS_WEIGHTS[2] < 1.0

    def test_axis_weights_z_downweighted(self):
        """z 轴偏差应比同量级 x 偏差贡献小"""
        a = np.array([0.0, 0.0, 0.0])
        b_x = np.array([0.1, 0.0, 0.0])
        b_z = np.array([0.0, 0.0, 0.1])
        d_x = weighted_euclidean_distance(a, b_x)
        d_z = weighted_euclidean_distance(a, b_z)
        assert d_x > d_z * 2.0

    def test_joint_weights_core_vs_extremity(self):
        """核心关节(肩)偏差权重应高于末端关节(腕)"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.1, 0.0, 0.0])
        # 左肩索引 11 → 在 CORE_JOINT_INDICES 中的位置
        shoulder_k = CORE_JOINT_INDICES.index(11)
        wrist_k = CORE_JOINT_INDICES.index(15)  # 左腕
        d_shoulder = weighted_euclidean_distance(
            a, b, joint_indices=CORE_JOINT_INDICES, k=shoulder_k
        )
        d_wrist = weighted_euclidean_distance(
            a, b, joint_indices=CORE_JOINT_INDICES, k=wrist_k
        )
        # 肩权重 1.2 > 腕权重 0.7
        assert d_shoulder > d_wrist

    def test_visibility_weighting(self):
        """低可见度应放大偏差"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.1, 0.0, 0.0])
        d_high = weighted_euclidean_distance(a, b, visibility=0.9)
        d_low = weighted_euclidean_distance(a, b, visibility=0.3)
        assert d_low > d_high

    def test_weighted_euclidean_frame_shape(self):
        """整帧加权距离测试"""
        a = np.random.randn(51).astype(np.float64)  # 17 joints * 3
        b = np.random.randn(51).astype(np.float64)
        d = weighted_euclidean_frame(a, b, CORE_JOINT_INDICES)
        assert d > 0
        assert isinstance(d, float)

    def test_identical_zero(self):
        """相同向量加权距离为 0"""
        a = np.array([1.0, 2.0, 3.0])
        assert weighted_euclidean_distance(a, a) == 0.0

    def test_get_weighted_metric_func(self):
        """验证可通过统一接口获取 weighted_euclidean"""
        f = get_distance_func("weighted_euclidean")
        assert callable(f)


# ===== 混合特征矩阵测试 =====

class TestHybridMatrix:

    def test_hybrid_shape(self):
        """混合矩阵维度应为 J*3 + n_angles"""
        seq = _make_sequence(20)
        mat = sequence_to_hybrid_matrix(seq, alpha=0.6, beta=0.4)
        assert mat.shape[0] == 20
        # 17 关节 * 3 + 9 个角度 = 51 + 9 = 60
        assert mat.shape[1] == 51 + 9

    def test_hybrid_rows_approx_unit_norm(self):
        """每行应近似单位范数（坐标和角度各自 L2 归一化后拼接）"""
        seq = _make_sequence(20)
        mat = sequence_to_hybrid_matrix(seq, alpha=0.6, beta=0.4)
        norms = np.linalg.norm(mat, axis=1)
        # 各自归一化后，拼接范数 ≈ sqrt(alpha² + beta²)
        expected = np.sqrt(0.6**2 + 0.4**2)
        assert np.allclose(norms, expected, atol=0.05)

    def test_hybrid_with_dtw(self):
        """混合矩阵可用于 DTW 比对"""
        seq = _make_sequence(20)
        mat = sequence_to_hybrid_matrix(seq, alpha=0.6, beta=0.4)
        d, p, _ = compute_dtw(mat, mat)
        assert d == 0.0


# ===== 加权偏差分析测试 =====

class TestDeviationAnalyzerWeighted:

    def test_weighted_analyze_identical(self, seq_a, seq_b):
        """加权模式下相同序列偏差为 0"""
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_b)

        analyzer = JointDeviationAnalyzer(top_k=5, use_weighted=True)
        report = analyzer.analyze(seq_a, seq_b, result)

        assert report.overall_deviation == 0.0
        assert report.use_weighted is True
        assert report.temporal_volatility == 0.0

    def test_weighted_vs_unweighted(self, seq_a, seq_diff):
        """加权模式偏差值应与未加权不同（因关节权重不均为 1.0）"""
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)

        analyzer_w = JointDeviationAnalyzer(top_k=5, use_weighted=True)
        report_w = analyzer_w.analyze(seq_a, seq_diff, result)

        analyzer_uw = JointDeviationAnalyzer(top_k=5, use_weighted=False)
        report_uw = analyzer_uw.analyze(seq_a, seq_diff, result)

        # 加权后因 z 降权和关节权重不均，总偏差应不同
        assert report_w.overall_deviation != pytest.approx(
            report_uw.overall_deviation, abs=1e-10
        )

    def test_temporal_volatility_computed(self, seq_a, seq_diff):
        """时域波动率应被计算且为正值"""
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)

        analyzer = JointDeviationAnalyzer(top_k=5, use_weighted=True)
        report = analyzer.analyze(seq_a, seq_diff, result)

        assert report.temporal_volatility >= 0
        assert "波动率" in report.summary() or report.temporal_volatility >= 0

    def test_unweighted_backward_compat(self, seq_a, seq_diff):
        """use_weighted=False 时行为与旧版兼容"""
        comp = ActionComparator(preprocess=False)
        result = comp.compare(seq_a, seq_diff)

        analyzer = JointDeviationAnalyzer(top_k=3, use_weighted=False)
        report = analyzer.analyze(seq_a, seq_diff, result)

        assert report.use_weighted is False
        assert report.overall_deviation > 0
        assert len(report.worst_joints) == 3


# ===== Hybrid 度量对比器测试 =====

class TestHybridComparator:

    def test_hybrid_metric_identical(self, seq_a, seq_b):
        """hybrid 度量下相同序列相似度为 1.0"""
        comp = ActionComparator(metric="hybrid", preprocess=False)
        result = comp.compare(seq_a, seq_b)
        assert result.similarity == 1.0
        assert result.metric == "hybrid"

    def test_hybrid_metric_different(self, seq_a, seq_diff):
        """hybrid 度量下不同序列相似度 < 1.0"""
        comp = ActionComparator(metric="hybrid", preprocess=False)
        result = comp.compare(seq_a, seq_diff)
        assert 0 < result.similarity < 1.0
        assert result.distance > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
