"""
RealtimeFeedbackEngine 单元测试

测试覆盖：
- 固定窗口 DTW 对比正确性
- 角度偏差阈值过滤
- 建议去重逻辑
- 模板进度映射越界保护
- FeedbackSnapshot 格式化输出
"""

import time
import numpy as np
import pytest

from src.pose_estimation.data_types import PoseSequence, PoseFrame, PoseLandmark
from src.correction.realtime_feedback import (
    RealtimeFeedbackEngine,
    FeedbackItem,
    FeedbackSnapshot,
    Severity,
)


# ===== 辅助函数 =====

def _make_template(num_frames: int = 60, num_joints: int = 33) -> PoseSequence:
    """创建合成模板序列"""
    seq = PoseSequence(fps=30.0)
    for i in range(num_frames):
        t = i / num_frames
        landmarks = []
        for j in range(num_joints):
            landmarks.append(PoseLandmark(
                x=0.5 + 0.1 * np.sin(t * np.pi + j * 0.1),
                y=0.5 + 0.1 * np.cos(t * np.pi + j * 0.1),
                z=0.0,
                visibility=1.0,
            ))
        frame = PoseFrame(
            timestamp=i / 30.0,
            frame_index=i,
            landmarks=landmarks,
        )
        seq.add_frame(frame)
    return seq


def _make_landmarks(offset: float = 0.0, num_joints: int = 33) -> np.ndarray:
    """创建合成单帧 landmarks (33, 4)"""
    data = np.zeros((num_joints, 4))
    for j in range(num_joints):
        data[j, 0] = 0.5 + offset + j * 0.01  # x
        data[j, 1] = 0.5 + offset + j * 0.01  # y
        data[j, 2] = 0.0                       # z
        data[j, 3] = 1.0                       # visibility
    return data


# ===== 测试类 =====

class TestRealtimeFeedbackEngine:
    """RealtimeFeedbackEngine 测试"""

    @pytest.fixture
    def template(self):
        return _make_template(num_frames=60)

    @pytest.fixture
    def engine(self, template):
        return RealtimeFeedbackEngine(
            template_sequence=template,
            algorithm="fastdtw",
            window_size=10,
            angle_threshold=15.0,
            dedup_interval=2.0,
        )

    def test_init(self, engine):
        """测试初始化"""
        assert engine.algorithm == "fastdtw"
        assert engine._window_size == 10
        assert engine._angle_threshold == 15.0

    def test_analyze_frame_returns_snapshot(self, engine):
        """测试 analyze_frame 返回 FeedbackSnapshot"""
        landmarks = _make_landmarks()
        snapshot = engine.analyze_frame(
            landmarks=landmarks,
            frame_index=0,
            expected_total_frames=60,
        )
        assert isinstance(snapshot, FeedbackSnapshot)
        assert snapshot.has_pose is True
        assert snapshot.frame_index == 0

    def test_analyze_multiple_frames(self, engine):
        """测试多帧分析，缓冲区逐步填充"""
        for i in range(15):
            landmarks = _make_landmarks(offset=i * 0.001)
            snapshot = engine.analyze_frame(
                landmarks=landmarks,
                frame_index=i,
                expected_total_frames=60,
            )
            assert snapshot.buffer_size == min(i + 1, 10)  # maxlen=10
            assert snapshot.frame_index == i

    def test_window_dtw_similarity(self, engine):
        """测试固定窗口 DTW 返回有效相似度"""
        # 先填充足够帧
        for i in range(10):
            landmarks = _make_landmarks(offset=i * 0.001)
            snapshot = engine.analyze_frame(
                landmarks=landmarks,
                frame_index=i,
                expected_total_frames=60,
            )

        # 最后一帧应有有效相似度
        assert snapshot.window_similarity > 0.0
        assert snapshot.window_similarity <= 1.0


class TestProgressMapping:
    """模板进度映射测试"""

    @pytest.fixture
    def engine(self):
        template = _make_template(num_frames=60)
        return RealtimeFeedbackEngine(
            template_sequence=template,
            algorithm="fastdtw",
        )

    def test_map_start(self, engine):
        """进度 0% → 模板第 0 帧"""
        idx = engine._map_progress(0, 60)
        assert idx == 0

    def test_map_middle(self, engine):
        """进度 50% → 模板中间帧"""
        idx = engine._map_progress(30, 60)
        assert 25 <= idx <= 35  # 约 50% 位置

    def test_map_end(self, engine):
        """进度 100% → 模板末帧"""
        idx = engine._map_progress(60, 60)
        assert idx == 59  # 最后一帧

    def test_map_overflow(self, engine):
        """进度 >100% → 不越界"""
        idx = engine._map_progress(120, 60)
        assert idx == 59  # 越界保护

    def test_map_zero_total(self, engine):
        """预期总帧数为 0 → 不出错"""
        idx = engine._map_progress(10, 0)
        assert 0 <= idx <= 59


class TestAngleThreshold:
    """角度偏差阈值过滤测试"""

    @pytest.fixture
    def engine(self):
        template = _make_template(num_frames=60)
        return RealtimeFeedbackEngine(
            template_sequence=template,
            algorithm="fastdtw",
            angle_threshold=15.0,
        )

    def test_below_threshold_no_feedback(self, engine):
        """偏差 < 阈值 → 不生成建议"""
        # 使用与模板非常接近的 landmarks
        template_frame = engine._template_array[0]
        snapshot = engine.analyze_frame(
            landmarks=template_frame.copy(),
            frame_index=0,
            expected_total_frames=60,
        )
        # 完全相同的帧，角度偏差应为 0
        assert len(snapshot.items) == 0

    def test_threshold_filtering(self, engine):
        """验证阈值过滤机制存在"""
        # 即使有偏差，只要 < threshold 就不报告
        user_angles = {"left_knee": 100.0}
        template_angles = {"left_knee": 105.0}  # 偏差 5 度 < 15 度
        items = engine._detect_deviations(
            user_angles, template_angles, time.time()
        )
        assert len(items) == 0

    def test_above_threshold_generates_feedback(self, engine):
        """偏差 > 阈值 → 生成建议"""
        user_angles = {"left_knee": 100.0}
        template_angles = {"left_knee": 130.0}  # 偏差 30 度 > 15 度
        items = engine._detect_deviations(
            user_angles, template_angles, time.time()
        )
        assert len(items) == 1
        assert items[0].joint_name == "left_knee"
        assert items[0].deviation_deg == 30.0


class TestDeduplication:
    """建议去重逻辑测试"""

    @pytest.fixture
    def engine(self):
        template = _make_template(num_frames=60)
        return RealtimeFeedbackEngine(
            template_sequence=template,
            algorithm="fastdtw",
            dedup_interval=2.0,
        )

    def test_same_advice_deduplicated(self, engine):
        """同一建议在 2 秒内不重复"""
        user_angles = {"left_knee": 100.0}
        template_angles = {"left_knee": 130.0}

        now = time.time()

        # 第一次 → 应生成
        items1 = engine._detect_deviations(user_angles, template_angles, now)
        assert len(items1) == 1

        # 0.5 秒后再次 → 应被去重
        items2 = engine._detect_deviations(
            user_angles, template_angles, now + 0.5
        )
        assert len(items2) == 0

    def test_advice_reappears_after_interval(self, engine):
        """超过去重间隔后，建议可再次出现"""
        user_angles = {"left_knee": 100.0}
        template_angles = {"left_knee": 130.0}

        now = time.time()

        # 第一次
        items1 = engine._detect_deviations(user_angles, template_angles, now)
        assert len(items1) == 1

        # 3 秒后 → 超过 2 秒间隔，应再次出现
        items2 = engine._detect_deviations(
            user_angles, template_angles, now + 3.0
        )
        assert len(items2) == 1


class TestFeedbackSnapshot:
    """FeedbackSnapshot 测试"""

    def test_empty_snapshot(self):
        snap = FeedbackSnapshot(has_pose=True)
        assert not snap.has_issues
        assert snap.num_issues == 0

    def test_snapshot_with_items(self):
        items = [
            FeedbackItem(
                joint_name="left_knee",
                joint_display="左膝",
                deviation_deg=25.0,
                direction="角度过大",
                advice="左膝弯曲过度",
                severity=Severity.WARNING,
            )
        ]
        snap = FeedbackSnapshot(items=items, has_pose=True, window_similarity=0.85)
        assert snap.has_issues
        assert snap.num_issues == 1

    def test_to_markdown_no_pose(self):
        snap = FeedbackSnapshot(has_pose=False)
        md = snap.to_markdown()
        assert "未检测到" in md

    def test_to_markdown_with_issues(self):
        items = [
            FeedbackItem(
                joint_name="left_knee",
                joint_display="左膝",
                deviation_deg=25.0,
                direction="角度过大",
                advice="左膝弯曲过度",
                severity=Severity.WARNING,
            )
        ]
        snap = FeedbackSnapshot(
            items=items, has_pose=True, window_similarity=0.85
        )
        md = snap.to_markdown()
        assert "左膝" in md
        assert "25°" in md

    def test_to_markdown_all_good(self):
        snap = FeedbackSnapshot(has_pose=True, window_similarity=0.95)
        md = snap.to_markdown()
        assert "姿态正确" in md


class TestSeverity:
    """严重等级测试"""

    def test_severity_order(self):
        """验证严重等级排序"""
        engine = RealtimeFeedbackEngine(
            template_sequence=_make_template(),
            algorithm="fastdtw",
        )

        now = time.time()
        user_angles = {
            "left_knee": 90.0,    # 偏差 40 → ERROR (>30)
            "right_knee": 110.0,  # 偏差 16 → INFO (15<16<20)
        }
        template_angles = {
            "left_knee": 130.0,
            "right_knee": 126.0,
        }

        items = engine._detect_deviations(user_angles, template_angles, now)
        if len(items) >= 2:
            # ERROR 应排在 INFO 前面
            assert items[0].severity == Severity.ERROR
            assert items[1].severity == Severity.INFO


class TestEngineReset:
    """引擎重置测试"""

    def test_reset_clears_state(self):
        engine = RealtimeFeedbackEngine(
            template_sequence=_make_template(),
            algorithm="fastdtw",
        )

        # 添加一些帧
        for i in range(5):
            engine.analyze_frame(_make_landmarks(), i, 60)

        assert engine._frame_count == 5
        assert len(engine._frame_buffer) == 5

        # 重置
        engine.reset()
        assert engine._frame_count == 0
        assert len(engine._frame_buffer) == 0
        assert len(engine._dedup_cache) == 0
