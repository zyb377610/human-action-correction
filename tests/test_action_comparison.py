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
        """默认使用 17 个核心关节点，输出 (T, 51)"""
        matrix = sequence_to_feature_matrix(seq_a)
        assert matrix.shape == (20, 51)  # 17 核心关节 × 3

    def test_sequence_to_feature_matrix_all_joints(self, seq_a):
        """使用全部 33 个关节点时，输出 (T, 99)"""
        all_joints = list(range(33))
        matrix = sequence_to_feature_matrix(seq_a, joint_indices=all_joints)
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
        assert result.cost_matrix is None

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
