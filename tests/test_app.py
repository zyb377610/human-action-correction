"""
应用层模块单元测试

覆盖: AnalysisResult, ProcessedFrame, SessionManager, AppPipeline 基础功能
"""

import json
import pytest
import numpy as np

from src.app.data_types import AnalysisResult, ProcessedFrame
from src.app.session import SessionManager, RecordingState
from src.correction.data_types import CorrectionItem, CorrectionReport, PRIORITY_HIGH


# ===== AnalysisResult 测试 =====

class TestAnalysisResult:
    def test_default_values(self):
        result = AnalysisResult()
        assert result.action_name == ""
        assert result.quality_score == 0.0
        assert result.num_corrections == 0
        assert not result.has_issues

    def test_with_corrections(self):
        items = [
            CorrectionItem("a", "A", 0.1, "desc", "advice", PRIORITY_HIGH),
            CorrectionItem("b", "B", 0.2, "desc", "advice", PRIORITY_HIGH),
        ]
        result = AnalysisResult(
            action_name="squat",
            action_display_name="深蹲",
            quality_score=75.0,
            similarity=0.8,
            corrections=items,
        )
        assert result.num_corrections == 2
        assert result.has_issues
        assert result.action_display_name == "深蹲"

    def test_summary(self):
        result = AnalysisResult(
            action_name="squat",
            action_display_name="深蹲",
            quality_score=85.0,
            similarity=0.9,
        )
        s = result.summary()
        assert "深蹲" in s
        assert "85.0" in s
        assert "90.0%" in s

    def test_summary_with_confidence(self):
        result = AnalysisResult(
            action_name="squat",
            quality_score=85.0,
            similarity=0.9,
            confidence=0.95,
        )
        s = result.summary()
        assert "95.0%" in s


# ===== ProcessedFrame 测试 =====

class TestProcessedFrame:
    def test_no_pose(self):
        frame = ProcessedFrame()
        assert not frame.has_pose

    def test_with_pose(self):
        landmarks = np.random.rand(33, 4).astype(np.float32)
        frame = ProcessedFrame(
            annotated_image=np.zeros((480, 640, 3), dtype=np.uint8),
            landmarks=landmarks,
        )
        assert frame.has_pose
        assert frame.landmarks.shape == (33, 4)


# ===== SessionManager 测试 =====

class TestSessionManager:
    def test_initial_state(self):
        sm = SessionManager()
        assert sm.state == RecordingState.IDLE
        assert sm.is_idle
        assert not sm.is_recording
        assert sm.frame_count == 0

    def test_start_recording(self):
        sm = SessionManager()
        sm.start_recording()
        assert sm.state == RecordingState.RECORDING
        assert sm.is_recording

    def test_add_frame_when_recording(self):
        sm = SessionManager()
        sm.start_recording()
        for i in range(10):
            sm.add_frame(np.random.rand(33, 4))
        assert sm.frame_count == 10

    def test_add_frame_when_idle_ignored(self):
        sm = SessionManager()
        sm.add_frame(np.random.rand(33, 4))
        assert sm.frame_count == 0

    def test_stop_recording_sufficient_frames(self):
        sm = SessionManager()
        sm.start_recording()
        for i in range(10):
            sm.add_frame(np.random.rand(33, 4))
        seq = sm.stop_recording()
        assert seq is not None
        assert seq.shape == (10, 33, 4)
        assert sm.state == RecordingState.ANALYZING

    def test_stop_recording_insufficient_frames(self):
        sm = SessionManager()
        sm.start_recording()
        sm.add_frame(np.random.rand(33, 4))
        seq = sm.stop_recording()
        assert seq is None
        assert sm.state == RecordingState.IDLE

    def test_finish_analysis(self):
        sm = SessionManager()
        sm.start_recording()
        for i in range(10):
            sm.add_frame(np.random.rand(33, 4))
        sm.stop_recording()
        result = AnalysisResult(action_name="test", quality_score=80.0)
        sm.finish_analysis(result)
        assert sm.is_idle
        assert sm.last_result is not None
        assert sm.last_result.action_name == "test"
        assert sm.frame_count == 0

    def test_reset(self):
        sm = SessionManager()
        sm.start_recording()
        sm.add_frame(np.random.rand(33, 4))
        sm.reset()
        assert sm.is_idle
        assert sm.frame_count == 0
        assert sm.last_result is None

    def test_action_choices(self):
        sm = SessionManager()
        choices = sm.get_action_choices()
        assert choices[0] == "自动识别"
        assert isinstance(choices, list)

    def test_state_display(self):
        sm = SessionManager()
        assert "空闲" in sm.state_display
        sm.start_recording()
        assert "录制中" in sm.state_display

    def test_get_action_list_empty(self):
        sm = SessionManager()
        assert sm.get_action_list() == []
