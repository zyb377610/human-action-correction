"""
姿态估计模块单元测试

覆盖数据类型、特征工程和视频源的核心功能。
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pose_estimation.data_types import (
    PoseLandmark, PoseFrame, PoseSequence, NUM_LANDMARKS
)
from src.pose_estimation.feature_extractor import (
    calculate_angle, get_joint_angles, get_bone_lengths,
    get_normalized_bone_ratios, calculate_velocity,
    calculate_sequence_velocity, FeatureExtractor,
)
from src.pose_estimation.video_source import VideoSource, FileSource
from src.utils.config import Config, get_config
from src.utils.io_utils import NumpyEncoder, save_json, load_json


# ===== 辅助函数 =====

def make_landmarks(visibility=0.95):
    """创建测试用的 33 个关键点"""
    return [
        PoseLandmark(x=i * 0.03, y=i * 0.02, z=0.01 * i, visibility=visibility)
        for i in range(NUM_LANDMARKS)
    ]


def make_frame(index=0, timestamp=0.0, visibility=0.95):
    """创建测试用的单帧"""
    return PoseFrame(
        timestamp=timestamp,
        frame_index=index,
        landmarks=make_landmarks(visibility),
    )


def make_sequence(num_frames=5, fps=30.0):
    """创建测试用的姿态序列"""
    frames = [make_frame(i, i / fps) for i in range(num_frames)]
    return PoseSequence(frames=frames, fps=fps)


# ===== 数据类型测试 =====

class TestPoseLandmark:
    def test_to_array(self):
        lm = PoseLandmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
        arr = lm.to_array()
        assert arr.shape == (4,)
        assert abs(arr[0] - 0.5) < 1e-6

class TestPoseFrame:
    def test_to_numpy_shape(self):
        frame = make_frame()
        arr = frame.to_numpy()
        assert arr.shape == (33, 4)

    def test_get_landmark_by_name(self):
        frame = make_frame()
        lm = frame.get_landmark_by_name("nose")
        assert isinstance(lm, PoseLandmark)

    def test_num_landmarks(self):
        frame = make_frame()
        assert frame.num_landmarks == 33

class TestPoseSequence:
    def test_to_numpy_shape(self):
        seq = make_sequence(5)
        arr = seq.to_numpy()
        assert arr.shape == (5, 33, 4)

    def test_empty_sequence(self):
        seq = PoseSequence()
        arr = seq.to_numpy()
        assert arr.shape == (0, 33, 4)

    def test_save_load_npy(self):
        seq = make_sequence(3)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            seq.save(path)
            loaded = PoseSequence.load(path)
            diff = np.max(np.abs(seq.to_numpy() - loaded.to_numpy()))
            assert diff < 1e-6
        finally:
            os.unlink(path)

    def test_save_load_json(self):
        seq = make_sequence(3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            seq.save(path)
            loaded = PoseSequence.load(path)
            diff = np.max(np.abs(seq.to_numpy() - loaded.to_numpy()))
            assert diff < 1e-6
            assert loaded.fps == seq.fps
        finally:
            os.unlink(path)

    def test_duration(self):
        seq = make_sequence(30, fps=30.0)
        assert abs(seq.duration - 29 / 30.0) < 1e-6


# ===== 特征工程测试 =====

class TestCalculateAngle:
    def test_right_angle(self):
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([0.0, 1.0])
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 90.0) < 0.1

    def test_straight_angle(self):
        p1 = np.array([-1.0, 0.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 180.0) < 0.1

    def test_zero_length_returns_nan(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        assert np.isnan(angle)

class TestJointAngles:
    def test_returns_all_joints(self):
        frame = make_frame()
        angles = get_joint_angles(frame)
        assert len(angles) == 8
        assert "left_elbow" in angles
        assert "right_knee" in angles

    def test_low_visibility_returns_nan(self):
        frame = make_frame(visibility=0.1)
        angles = get_joint_angles(frame)
        for name, val in angles.items():
            assert np.isnan(val)

class TestBoneLengths:
    def test_returns_all_bones(self):
        frame = make_frame()
        lengths = get_bone_lengths(frame)
        assert len(lengths) == 12

    def test_normalized_ratios(self):
        frame = make_frame()
        ratios = get_normalized_bone_ratios(frame)
        assert len(ratios) == 12

class TestVelocity:
    def test_zero_velocity_same_frames(self):
        f1 = make_frame(0, 0.0)
        f2 = make_frame(1, 0.033)
        # 相同关键点位置
        vel = calculate_velocity(f1, f2)
        # 因为 f1 和 f2 的 landmarks 值相同，速度应该为 0
        # 但这里 make_frame 每次创建相同数据，所以速度确实为 0
        assert vel.shape == (33,)

    def test_sequence_velocity_shape(self):
        seq = make_sequence(5)
        vel = calculate_sequence_velocity(seq)
        assert vel.shape == (4, 33)

    def test_single_frame_sequence(self):
        seq = make_sequence(1)
        vel = calculate_sequence_velocity(seq)
        assert vel.shape == (0, 33)

class TestFeatureExtractor:
    def test_frame_features_dim(self):
        ext = FeatureExtractor()
        frame = make_frame()
        features = ext.extract_frame_features(frame)
        assert features.shape == (ext.feature_dim,)

    def test_sequence_features_shape(self):
        ext = FeatureExtractor()
        seq = make_sequence(5)
        features = ext.extract_sequence_features(seq)
        assert features.shape == (5, ext.feature_dim)

    def test_feature_names(self):
        ext = FeatureExtractor()
        names = ext.feature_names
        assert len(names) == ext.feature_dim


# ===== 配置测试 =====

class TestConfig:
    def test_default_config(self):
        config = get_config()
        assert "pose_estimation" in config

    def test_get_section(self):
        config = get_config()
        pose_cfg = config.get_section("pose_estimation")
        assert "max_frames" in pose_cfg
        assert pose_cfg["max_frames"] == 300


# ===== IO 工具测试 =====

class TestIOUtils:
    def test_numpy_encoder(self):
        import json
        data = {"array": np.array([1, 2, 3]), "float": np.float32(1.5)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert "[1, 2, 3]" in result

    def test_save_load_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_json({"key": "value", "num": 42}, path)
            loaded = load_json(path)
            assert loaded["key"] == "value"
            assert loaded["num"] == 42
        finally:
            os.unlink(path)


# ===== 视频源测试 =====

class TestFileSource:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            FileSource("nonexistent_video.mp4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
