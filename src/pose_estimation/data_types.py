"""
姿态估计数据类型定义

定义姿态估计模块的核心数据结构：
- PoseLandmark: 单个关键点（坐标 + 可见度）
- PoseFrame: 单帧姿态（33个关键点 + 时间戳）
- PoseSequence: 姿态序列（多帧 + 帧率）
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, ClassVar

import numpy as np


# MediaPipe Pose 关键点数量
NUM_LANDMARKS = 33

# MediaPipe Pose 关键点名称（索引 0-32）
LANDMARK_NAMES = [
    "nose",                      # 0
    "left_eye_inner",            # 1
    "left_eye",                  # 2
    "left_eye_outer",            # 3
    "right_eye_inner",           # 4
    "right_eye",                 # 5
    "right_eye_outer",           # 6
    "left_ear",                  # 7
    "right_ear",                 # 8
    "mouth_left",                # 9
    "mouth_right",               # 10
    "left_shoulder",             # 11
    "right_shoulder",            # 12
    "left_elbow",                # 13
    "right_elbow",               # 14
    "left_wrist",                # 15
    "right_wrist",               # 16
    "left_pinky",                # 17
    "right_pinky",               # 18
    "left_index",                # 19
    "right_index",               # 20
    "left_thumb",                # 21
    "right_thumb",               # 22
    "left_hip",                  # 23
    "right_hip",                 # 24
    "left_knee",                 # 25
    "right_knee",                # 26
    "left_ankle",                # 27
    "right_ankle",               # 28
    "left_heel",                 # 29
    "right_heel",                # 30
    "left_foot_index",           # 31
    "right_foot_index",          # 32
]

# 骨骼连接关系（用于绘制骨骼图）
POSE_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7),      # 左眼 → 左耳
    (0, 4), (4, 5), (5, 6), (6, 8),      # 右眼 → 右耳
    (9, 10),                               # 嘴巴
    # 躯干
    (11, 12),                              # 双肩
    (11, 23), (12, 24),                    # 肩 → 髋
    (23, 24),                              # 双髋
    # 左臂
    (11, 13), (13, 15),                    # 左肩 → 左肘 → 左腕
    (15, 17), (15, 19), (15, 21),          # 左腕 → 手指
    (17, 19),
    # 右臂
    (12, 14), (14, 16),                    # 右肩 → 右肘 → 右腕
    (16, 18), (16, 20), (16, 22),          # 右腕 → 手指
    (18, 20),
    # 左腿
    (23, 25), (25, 27),                    # 左髋 → 左膝 → 左踝
    (27, 29), (27, 31), (29, 31),          # 左脚
    # 右腿
    (24, 26), (26, 28),                    # 右髋 → 右膝 → 右踝
    (28, 30), (28, 32), (30, 32),          # 右脚
]


@dataclass
class PoseLandmark:
    """
    单个人体关键点

    Attributes:
        x: 归一化 x 坐标 [0, 1]
        y: 归一化 y 坐标 [0, 1]
        z: 相对深度值
        visibility: 可见度 [0, 1]
    """
    x: float
    y: float
    z: float
    visibility: float

    def to_array(self) -> np.ndarray:
        """转为 (4,) 的 NumPy 数组 [x, y, z, visibility]"""
        return np.array([self.x, self.y, self.z, self.visibility],
                        dtype=np.float32)


@dataclass
class PoseFrame:
    """
    单帧姿态数据（33个关键点）

    Attributes:
        timestamp: 时间戳（秒）
        frame_index: 帧序号
        landmarks: 33 个关键点列表
    """
    timestamp: float
    frame_index: int
    landmarks: List[PoseLandmark]

    def to_numpy(self) -> np.ndarray:
        """
        转为 (33, 4) 的 NumPy 数组

        Returns:
            形状为 (33, 4) 的数组，每行为 [x, y, z, visibility]
        """
        return np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in self.landmarks],
            dtype=np.float32
        )

    def get_landmark(self, index: int) -> PoseLandmark:
        """
        按索引获取关键点

        Args:
            index: 关键点索引 (0-32)
        """
        return self.landmarks[index]

    def get_landmark_by_name(self, name: str) -> PoseLandmark:
        """
        按名称获取关键点

        Args:
            name: 关键点名称，如 "left_shoulder"
        """
        idx = LANDMARK_NAMES.index(name)
        return self.landmarks[idx]

    @property
    def num_landmarks(self) -> int:
        """关键点数量"""
        return len(self.landmarks)


@dataclass
class PoseSequence:
    """
    姿态序列（多帧关键点数据）

    Attributes:
        frames: PoseFrame 帧列表
        fps: 视频帧率
        metadata: 附加元信息
    """
    frames: List[PoseFrame] = field(default_factory=list)
    fps: float = 30.0
    metadata: dict = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        """
        转为 (T, 33, 4) 的 NumPy 数组

        Returns:
            形状为 (T, 33, 4) 的数组，T 为帧数
        """
        if len(self.frames) == 0:
            return np.empty((0, NUM_LANDMARKS, 4), dtype=np.float32)
        return np.stack([f.to_numpy() for f in self.frames], axis=0)

    def add_frame(self, frame: PoseFrame):
        """添加一帧"""
        self.frames.append(frame)

    @property
    def num_frames(self) -> int:
        """帧数"""
        return len(self.frames)

    @property
    def duration(self) -> float:
        """序列时长（秒）"""
        if len(self.frames) == 0:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    def save(self, path: str):
        """
        保存姿态序列到文件

        根据扩展名自动选择格式：
        - .npy: NumPy 二进制格式（仅关键点数据）
        - .json: JSON 格式（含完整元信息）

        Args:
            path: 保存文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".npy":
            np.save(str(path), self.to_numpy())
        elif path.suffix == ".json":
            data = {
                "fps": self.fps,
                "num_frames": self.num_frames,
                "metadata": self.metadata,
                "frames": [
                    {
                        "timestamp": f.timestamp,
                        "frame_index": f.frame_index,
                        "landmarks": [
                            {
                                "x": float(lm.x),
                                "y": float(lm.y),
                                "z": float(lm.z),
                                "visibility": float(lm.visibility),
                            }
                            for lm in f.landmarks
                        ],
                    }
                    for f in self.frames
                ],
            }
            with open(str(path), "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}，支持 .npy 和 .json")

    @classmethod
    def load(cls, path: str) -> "PoseSequence":
        """
        从文件加载姿态序列

        Args:
            path: 文件路径（.npy 或 .json）

        Returns:
            PoseSequence 实例
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if path.suffix == ".npy":
            arr = np.load(str(path))  # (T, 33, 4)
            seq = cls(fps=30.0)  # npy 格式不含帧率信息，使用默认值
            for i in range(arr.shape[0]):
                landmarks = [
                    PoseLandmark(
                        x=float(arr[i, j, 0]),
                        y=float(arr[i, j, 1]),
                        z=float(arr[i, j, 2]),
                        visibility=float(arr[i, j, 3]),
                    )
                    for j in range(arr.shape[1])
                ]
                frame = PoseFrame(
                    timestamp=i / seq.fps,
                    frame_index=i,
                    landmarks=landmarks,
                )
                seq.add_frame(frame)
            return seq

        elif path.suffix == ".json":
            with open(str(path), "r", encoding="utf-8") as fp:
                data = json.load(fp)
            seq = cls(
                fps=data.get("fps", 30.0),
                metadata=data.get("metadata", {}),
            )
            for f_data in data["frames"]:
                landmarks = [
                    PoseLandmark(**lm_data) for lm_data in f_data["landmarks"]
                ]
                frame = PoseFrame(
                    timestamp=f_data["timestamp"],
                    frame_index=f_data["frame_index"],
                    landmarks=landmarks,
                )
                seq.add_frame(frame)
            return seq

        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}，支持 .npy 和 .json")
