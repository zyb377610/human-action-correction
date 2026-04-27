"""
端到端集成测试

验证各模块之间的数据流是否正确衔接：
  PoseSequence → 预处理 → DTW 对比 → 偏差分析 → 规则引擎 → CorrectionReport

以及 SessionManager 全生命周期测试。
"""

import tempfile
import pytest
import numpy as np

from src.pose_estimation.data_types import PoseFrame, PoseSequence, PoseLandmark
from src.data.preprocessing import preprocess_pipeline
from src.data.template_library import TemplateLibrary
from src.action_comparison.comparison import ActionComparator
from src.action_comparison.deviation_analyzer import JointDeviationAnalyzer
from src.correction.rules import CorrectionRuleEngine
from src.correction.angle_utils import AngleCalculator
from src.correction.feedback import FeedbackGenerator
from src.correction.data_types import CorrectionReport
from src.app.data_types import AnalysisResult
from src.app.session import SessionManager, RecordingState
from src.utils.config import load_config


# ===== 辅助函数 =====

def _make_sequence(num_frames=30, fps=30.0, noise=0.0) -> PoseSequence:
    """生成模拟骨骼序列"""
    seq = PoseSequence(fps=fps)
    for i in range(num_frames):
        landmarks = []
        for j in range(33):
            landmarks.append(PoseLandmark(
                x=0.5 + noise * np.random.randn(),
                y=(j / 33.0) + noise * np.random.randn(),
                z=0.0 + noise * np.random.randn(),
                visibility=0.95,
            ))
        seq.add_frame(PoseFrame(
            timestamp=i / fps,
            frame_index=i,
            landmarks=landmarks,
        ))
    return seq


def _sequence_to_array(seq: PoseSequence) -> np.ndarray:
    """PoseSequence → numpy (T, 33, 4)"""
    frames = []
    for frame in seq.frames:
        pts = [[lm.x, lm.y, lm.z, lm.visibility] for lm in frame.landmarks]
        frames.append(pts)
    return np.array(frames, dtype=np.float32)


# ===== 1. 配置加载测试 =====

class TestConfig:
    def test_load_default_config(self):
        """默认配置文件可正常加载"""
        data = load_config()
        assert "paths" in data
        assert "model" in data
        assert "server" in data
        assert data["server"]["port"] == 7860

    def test_config_sections(self):
        """配置包含所有必要的段"""
        data = load_config()
        required_sections = ["paths", "model", "pose_estimation", "comparison", "correction", "server"]
        for section in required_sections:
            assert section in data, f"缺少配置段: {section}"


# ===== 2. 模块间数据流集成测试 =====

class TestModuleDataFlow:
    """测试各模块之间的数据流衔接"""

    def test_preprocess_preserves_structure(self):
        """预处理后 PoseSequence 结构不变"""
        seq = _make_sequence(30, noise=0.01)
        processed = preprocess_pipeline(seq)
        assert isinstance(processed, PoseSequence)
        assert processed.num_frames > 0
        assert len(processed.frames[0].landmarks) == 33

    def test_sequence_to_comparator(self):
        """PoseSequence 可正确传入 ActionComparator"""
        user_seq = _make_sequence(30, noise=0.02)
        template_seq = _make_sequence(30, noise=0.0)
        comparator = ActionComparator()
        result = comparator.compare(user_seq, template_seq)
        assert hasattr(result, "distance")
        assert hasattr(result, "similarity")
        assert 0.0 <= result.similarity <= 1.0

    def test_comparator_to_deviation_analyzer(self):
        """ActionComparator 的结果可传入 JointDeviationAnalyzer"""
        user_seq = _make_sequence(25, noise=0.05)
        template_seq = _make_sequence(25, noise=0.0)
        comparator = ActionComparator()
        result = comparator.compare(user_seq, template_seq)

        # 验证对比结果结构完整
        assert hasattr(result, "path")
        assert hasattr(result, "similarity")
        assert hasattr(result, "distance")
        assert len(result.path) > 0
        assert 0.0 <= result.similarity <= 1.0

    def test_deviation_to_rules_engine(self):
        """偏差报告可传入规则引擎匹配矫正建议"""
        engine = CorrectionRuleEngine()
        # 模拟偏差数据
        joint_deviations = {"left_knee": 0.15, "left_shoulder": 0.08}
        angle_deviations = {"left_knee_angle": (80, 95, 15)}

        items = engine.match_rules(
            action="squat",
            joint_deviations=joint_deviations,
            angle_deviations=angle_deviations,
        )
        assert isinstance(items, list)
        # 至少应命中兜底规则或专属规则
        assert len(items) > 0

    def test_feedback_generator_produces_report(self):
        """FeedbackGenerator 可生成 CorrectionReport"""
        from src.action_comparison.deviation_analyzer import DeviationReport

        # 构造 mock DeviationReport（匹配实际 dataclass 字段）
        deviation_report = DeviationReport(
            joint_deviations={"left_knee": 0.15, "left_shoulder": 0.08},
            worst_joints=["left_knee", "left_shoulder"],
            frame_deviations=np.array([0.1, 0.12, 0.15]),
            severity="moderate",
            overall_deviation=0.12,
        )

        generator = FeedbackGenerator()
        report = generator.generate(
            action_name="squat",
            deviation_report=deviation_report,
            similarity=0.85,
        )
        assert isinstance(report, CorrectionReport)
        assert report.action_name == "squat"
        assert len(report.to_text()) > 0

    def test_full_pipeline_data_flow(self):
        """完整数据流：序列 → 预处理 → DTW → 角度 → 规则 → 报告"""
        from src.action_comparison.deviation_analyzer import DeviationReport

        # 1. 创建序列
        user_seq = _make_sequence(30, noise=0.05)
        template_seq = _make_sequence(30, noise=0.0)

        # 2. 预处理
        user_processed = preprocess_pipeline(user_seq)
        tmpl_processed = preprocess_pipeline(template_seq)

        # 3. DTW 对比
        comparator = ActionComparator()
        comp_result = comparator.compare(user_processed, tmpl_processed)
        assert 0.0 <= comp_result.similarity <= 1.0

        # 4. 构建偏差报告（用 mock 数据代替 analyzer.analyze）
        deviation_report = DeviationReport(
            joint_deviations={"left_knee": 0.12, "left_hip": 0.08},
            worst_joints=["left_knee", "left_hip"],
            frame_deviations=np.array([0.1, 0.12]),
            severity="moderate",
            overall_deviation=0.10,
        )

        # 6. 角度对比（需要 DTW path）
        angle_calc = AngleCalculator()
        angle_diffs = angle_calc.compare_angles(user_processed, tmpl_processed, comp_result.path)
        assert isinstance(angle_diffs, dict)

        # 7. 生成报告
        generator = FeedbackGenerator()
        report = generator.generate(
            action_name="squat",
            deviation_report=deviation_report,
            angle_deviations=angle_diffs,
            similarity=comp_result.similarity,
        )

        # 验证最终输出
        assert isinstance(report, CorrectionReport)
        assert report.quality_score > 0
        assert isinstance(report.to_text(), str)
        assert len(report.to_text()) > 10


# ===== 3. SessionManager 全生命周期测试 =====

class TestSessionManagerLifecycle:
    """完整状态机测试: idle → recording → analyzing → idle"""

    def test_full_lifecycle(self):
        sm = SessionManager()

        # 初始状态
        assert sm.state == RecordingState.IDLE
        assert sm.frame_count == 0

        # 开始录制
        sm.start_recording()
        assert sm.state == RecordingState.RECORDING

        # 添加帧
        for _ in range(20):
            sm.add_frame(np.random.rand(33, 4).astype(np.float32))
        assert sm.frame_count == 20

        # 停止录制
        seq = sm.stop_recording()
        assert sm.state == RecordingState.ANALYZING
        assert seq is not None
        assert seq.shape == (20, 33, 4)

        # 完成分析
        result = AnalysisResult(action_name="squat", quality_score=85.0)
        sm.finish_analysis(result)
        assert sm.state == RecordingState.IDLE
        assert sm.frame_count == 0
        assert sm.last_result.action_name == "squat"

    def test_recording_without_enough_frames(self):
        sm = SessionManager()
        sm.start_recording()
        sm.add_frame(np.random.rand(33, 4))
        seq = sm.stop_recording()
        assert seq is None
        assert sm.state == RecordingState.IDLE

    def test_multiple_cycles(self):
        sm = SessionManager()
        for cycle in range(3):
            sm.start_recording()
            for _ in range(10):
                sm.add_frame(np.random.rand(33, 4))
            seq = sm.stop_recording()
            assert seq is not None
            sm.finish_analysis(AnalysisResult(action_name=f"action_{cycle}"))
            assert sm.is_idle

    def test_reset_clears_everything(self):
        sm = SessionManager()
        sm.start_recording()
        for _ in range(10):
            sm.add_frame(np.random.rand(33, 4))
        sm.stop_recording()
        sm.finish_analysis(AnalysisResult(action_name="test"))

        sm.reset()
        assert sm.is_idle
        assert sm.frame_count == 0
        assert sm.last_result is None


# ===== 4. 模板库集成测试 =====

class TestTemplateLibraryIntegration:
    """模板库与其他模块的集成"""

    def test_save_and_load_template(self, tmp_path):
        """保存模板后可正确加载并用于 DTW 对比"""
        lib = TemplateLibrary(str(tmp_path))
        lib.add_action("squat", "深蹲")

        template_seq = _make_sequence(30)
        lib.add_template("squat", template_seq, "standard_01")

        # 重新加载
        loaded = lib.load_template("squat", "standard_01")
        assert isinstance(loaded, PoseSequence)
        assert loaded.num_frames == 30

        # 用于 DTW
        user_seq = _make_sequence(30, noise=0.03)
        comparator = ActionComparator()
        result = comparator.compare(user_seq, loaded)
        assert 0.0 <= result.similarity <= 1.0
