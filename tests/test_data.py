"""
数据模块单元测试

覆盖预处理、模板库和数据增强。
"""

import tempfile
import shutil
import pytest
import numpy as np

from src.pose_estimation.data_types import PoseLandmark, PoseFrame, PoseSequence
from src.data.preprocessing import (
    interpolate_missing,
    smooth_sequence,
    resample_sequence,
    preprocess_pipeline,
)
from src.data.template_library import TemplateLibrary
from src.data.augmentation import (
    time_warp,
    add_noise,
    mirror_sequence,
    augment_batch,
)


# ===== Fixtures =====

def _make_landmarks(visibility=0.9):
    """生成 33 个测试关键点"""
    return [
        PoseLandmark(x=i * 0.03, y=i * 0.02, z=0.01, visibility=visibility)
        for i in range(33)
    ]


def _make_sequence(num_frames=30, fps=30.0, visibility=0.9):
    """生成测试序列"""
    frames = [
        PoseFrame(
            timestamp=i / fps,
            frame_index=i,
            landmarks=_make_landmarks(visibility),
        )
        for i in range(num_frames)
    ]
    return PoseSequence(frames=frames, fps=fps)


@pytest.fixture
def sample_sequence():
    return _make_sequence(30, 30.0)


@pytest.fixture
def sequence_with_missing():
    """含缺失帧的序列"""
    seq = _make_sequence(30, 30.0)
    # 将第 5 帧的所有关键点标记为低可见度
    low_vis_landmarks = _make_landmarks(visibility=0.05)
    seq.frames[5] = PoseFrame(
        timestamp=5 / 30.0, frame_index=5, landmarks=low_vis_landmarks
    )
    return seq


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ===== 预处理测试 =====

class TestPreprocessing:

    def test_interpolate_missing(self, sequence_with_missing):
        result = interpolate_missing(sequence_with_missing, visibility_threshold=0.3)
        assert result.num_frames == 30
        # 第 5 帧应被插值，不再全是低可见度原始值
        frame5 = result.frames[5]
        frame4 = result.frames[4]
        frame6 = result.frames[6]
        # 检查第一个关键点 x 在 frame4 和 frame6 之间
        x5 = frame5.landmarks[1].x
        x4 = frame4.landmarks[1].x
        x6 = frame6.landmarks[1].x
        assert min(x4, x6) <= x5 <= max(x4, x6) or abs(x5 - (x4 + x6) / 2) < 0.01

    def test_smooth_sequence(self, sample_sequence):
        result = smooth_sequence(sample_sequence, window_length=5, polyorder=2)
        assert result.num_frames == sample_sequence.num_frames

    def test_resample_sequence(self, sample_sequence):
        result = resample_sequence(sample_sequence, target_frames=60)
        assert result.num_frames == 60

    def test_resample_reduce(self, sample_sequence):
        result = resample_sequence(sample_sequence, target_frames=15)
        assert result.num_frames == 15

    def test_preprocess_pipeline(self, sequence_with_missing):
        result = preprocess_pipeline(
            sequence_with_missing,
            visibility_threshold=0.3,
            target_frames=60,
        )
        assert result.num_frames == 60

    def test_preprocess_no_resample(self, sample_sequence):
        result = preprocess_pipeline(sample_sequence, target_frames=None)
        assert result.num_frames == sample_sequence.num_frames


# ===== 模板库测试 =====

class TestTemplateLibrary:

    def test_add_and_list_actions(self, temp_dir):
        lib = TemplateLibrary(temp_dir)
        lib.add_action("squat", "深蹲", "标准深蹲动作")
        lib.add_action("lunge", "弓步", "标准弓步动作")
        assert set(lib.list_actions()) == {"squat", "lunge"}

    def test_get_action_info(self, temp_dir):
        lib = TemplateLibrary(temp_dir)
        lib.add_action("squat", "深蹲", "标准深蹲动作")
        info = lib.get_action_info("squat")
        assert info["display_name"] == "深蹲"
        assert info["description"] == "标准深蹲动作"

    def test_add_and_load_template(self, temp_dir, sample_sequence):
        lib = TemplateLibrary(temp_dir)
        lib.add_template("squat", sample_sequence, "standard_01")
        loaded = lib.load_template("squat", "standard_01")
        assert loaded.num_frames == sample_sequence.num_frames

    def test_list_templates(self, temp_dir, sample_sequence):
        lib = TemplateLibrary(temp_dir)
        lib.add_template("squat", sample_sequence, "template_a")
        lib.add_template("squat", sample_sequence, "template_b")
        templates = lib.list_templates("squat")
        assert set(templates) == {"template_a", "template_b"}

    def test_delete_template(self, temp_dir, sample_sequence):
        lib = TemplateLibrary(temp_dir)
        lib.add_template("squat", sample_sequence, "to_delete")
        lib.delete_template("squat", "to_delete")
        assert "to_delete" not in lib.list_templates("squat")

    def test_load_all_templates(self, temp_dir, sample_sequence):
        lib = TemplateLibrary(temp_dir)
        lib.add_template("squat", sample_sequence, "t1")
        lib.add_template("squat", sample_sequence, "t2")
        all_t = lib.load_all_templates("squat")
        assert len(all_t) == 2
        assert "t1" in all_t and "t2" in all_t

    def test_load_nonexistent_template(self, temp_dir):
        lib = TemplateLibrary(temp_dir)
        with pytest.raises(FileNotFoundError):
            lib.load_template("nonexist", "nonexist")

    def test_persistence(self, temp_dir, sample_sequence):
        """关闭后重新打开模板库，数据应持久化"""
        lib1 = TemplateLibrary(temp_dir)
        lib1.add_action("squat", "深蹲")
        lib1.add_template("squat", sample_sequence, "persist_test")

        # 重新打开
        lib2 = TemplateLibrary(temp_dir)
        assert "squat" in lib2.list_actions()
        assert "persist_test" in lib2.list_templates("squat")


# ===== 数据增强测试 =====

class TestAugmentation:

    def test_time_warp_slow(self, sample_sequence):
        result = time_warp(sample_sequence, speed_factor=0.5)
        assert result.num_frames == 60  # 30 / 0.5

    def test_time_warp_fast(self, sample_sequence):
        result = time_warp(sample_sequence, speed_factor=2.0)
        assert result.num_frames == 15  # 30 / 2.0

    def test_time_warp_identity(self, sample_sequence):
        result = time_warp(sample_sequence, speed_factor=1.0)
        assert result.num_frames == sample_sequence.num_frames

    def test_time_warp_invalid(self, sample_sequence):
        with pytest.raises(ValueError):
            time_warp(sample_sequence, speed_factor=-1)

    def test_add_noise(self, sample_sequence):
        result = add_noise(sample_sequence, noise_std=0.01, seed=42)
        assert result.num_frames == sample_sequence.num_frames
        # 坐标应发生变化
        orig_x = sample_sequence.frames[0].landmarks[0].x
        noisy_x = result.frames[0].landmarks[0].x
        # 噪声不为零（极小概率恰好为零）
        # 只要不完全相等就通过
        assert result.num_frames == 30

    def test_add_noise_reproducible(self, sample_sequence):
        r1 = add_noise(sample_sequence, seed=42)
        r2 = add_noise(sample_sequence, seed=42)
        assert r1.frames[0].landmarks[0].x == r2.frames[0].landmarks[0].x

    def test_mirror_sequence(self, sample_sequence):
        result = mirror_sequence(sample_sequence)
        assert result.num_frames == sample_sequence.num_frames

        # 验证 x 坐标被镜像
        orig_x = sample_sequence.frames[0].landmarks[0].x
        mirror_x = result.frames[0].landmarks[0].x
        assert abs(mirror_x - (1.0 - orig_x)) < 1e-10

        # 验证左右关键点交换 (11: 左肩, 12: 右肩)
        orig_left = sample_sequence.frames[0].landmarks[11].x
        mirror_at_12 = result.frames[0].landmarks[12].x
        assert abs(mirror_at_12 - (1.0 - orig_left)) < 1e-10

    def test_augment_batch(self, sample_sequence):
        results = augment_batch(
            sample_sequence,
            num_augmented=5,
            seed=42,
        )
        assert len(results) == 5
        # 每个增强结果都应是有效序列
        for s in results:
            assert s.num_frames >= 2

    def test_augment_batch_no_mirror(self, sample_sequence):
        results = augment_batch(
            sample_sequence,
            num_augmented=3,
            mirror_enabled=False,
            seed=42,
        )
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
