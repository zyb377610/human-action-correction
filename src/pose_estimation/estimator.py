"""
姿态估计核心模块

封装 MediaPipe PoseLandmarker (Tasks API)，提供统一的姿态估计接口。
支持单帧图像和视频流两种输入模式，输出标准化的 PoseFrame/PoseSequence。
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from .data_types import PoseLandmark, PoseFrame, PoseSequence, NUM_LANDMARKS
from .video_source import VideoSource
from ..utils.config import get_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 模型复杂度与模型文件的映射
_MODEL_FILES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

# 默认模型目录
_DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


class PoseEstimator:
    """
    人体姿态估计器

    封装 MediaPipe PoseLandmarker，从图像/视频中提取 33 个人体骨骼关键点。

    使用示例:
        # 从配置文件初始化
        estimator = PoseEstimator()

        # 自定义参数初始化
        estimator = PoseEstimator(model_complexity=2, min_detection_confidence=0.7)

        # 单帧估计
        frame = cv2.imread("test.jpg")
        pose = estimator.estimate_frame(frame)

        # 视频流估计
        from src.pose_estimation.video_source import FileSource
        with FileSource("video.mp4") as source:
            sequence = estimator.estimate_video(source)
    """

    def __init__(
        self,
        model_complexity: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        初始化姿态估计器

        优先使用显式参数，未指定时从配置文件读取。

        Args:
            model_complexity: 模型复杂度 (0=Lite, 1=Full, 2=Heavy)
            min_detection_confidence: 最小检测置信度 [0, 1]
            min_tracking_confidence: 最小追踪置信度 [0, 1]
            model_path: 模型文件路径，None 时根据 model_complexity 自动选择
            config_path: 配置文件路径，None 时使用默认配置
        """
        # 从配置文件加载默认值
        config = get_config(config_path)
        pose_cfg = config.get_section("pose_estimation")

        # 合并参数（显式参数优先）
        self._model_complexity = (
            model_complexity if model_complexity is not None
            else pose_cfg.get("model_complexity", 1)
        )
        self._min_detection_confidence = (
            min_detection_confidence if min_detection_confidence is not None
            else pose_cfg.get("min_detection_confidence", 0.5)
        )
        self._min_tracking_confidence = (
            min_tracking_confidence if min_tracking_confidence is not None
            else pose_cfg.get("min_tracking_confidence", 0.5)
        )

        # 确定模型文件路径
        if model_path is not None:
            self._model_path = Path(model_path)
        else:
            model_file = _MODEL_FILES.get(self._model_complexity,
                                          _MODEL_FILES[1])
            self._model_path = _DEFAULT_MODEL_DIR / model_file

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {self._model_path}\n"
                f"请下载模型文件到 models/ 目录，或指定 model_path 参数。\n"
                f"下载地址: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models"
            )

        # 初始化 MediaPipe PoseLandmarker（图像模式）
        self._landmarker = None
        self._init_landmarker()

        logger.info(
            f"PoseEstimator 初始化完成: complexity={self._model_complexity}, "
            f"det_conf={self._min_detection_confidence}, "
            f"track_conf={self._min_tracking_confidence}"
        )

    def _init_landmarker(self):
        """初始化 MediaPipe PoseLandmarker"""
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # MediaPipe C++ 底层可能不支持中文路径，改用读取二进制数据方式加载
        with open(str(self._model_path), "rb") as f:
            model_data = f.read()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=model_data
            ),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            num_poses=1,  # 单人检测
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def estimate_frame(
        self,
        image: np.ndarray,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> Optional[PoseFrame]:
        """
        对单帧图像进行姿态估计

        Args:
            image: BGR 格式的图像 (NumPy 数组)
            timestamp: 时间戳（秒）
            frame_index: 帧序号

        Returns:
            PoseFrame 对象，未检测到人体时返回 None
        """
        if image is None or image.size == 0:
            return None

        # BGR → RGB（MediaPipe 要求 RGB 输入）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建 MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # 执行检测
        result = self._landmarker.detect(mp_image)

        # 检查是否检测到姿态
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # 取第一个检测到的人（单人模式）
        pose_landmarks = result.pose_landmarks[0]

        if len(pose_landmarks) < NUM_LANDMARKS:
            logger.warning(
                f"检测到的关键点数量不足: {len(pose_landmarks)} < {NUM_LANDMARKS}"
            )
            return None

        # 转换为 PoseFrame
        landmarks = []
        for lm in pose_landmarks:
            landmarks.append(PoseLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if lm.visibility is not None else 0.0,
            ))

        return PoseFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            landmarks=landmarks,
        )

    def estimate_video(
        self,
        source: VideoSource,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
    ) -> PoseSequence:
        """
        对视频流进行连续姿态估计

        Args:
            source: 视频源（CameraSource 或 FileSource）
            max_frames: 最大处理帧数，None 表示处理全部
            show_progress: 是否显示进度条

        Returns:
            PoseSequence 姿态序列
        """
        sequence = PoseSequence(fps=source.fps)
        frame_index = 0

        # 进度条设置
        total = None
        if hasattr(source, 'total_frames'):
            total = source.total_frames
        if max_frames is not None:
            total = min(total, max_frames) if total else max_frames

        pbar = tqdm(total=total, desc="姿态估计", disable=not show_progress)

        while True:
            success, frame = source.read()
            if not success:
                break

            if max_frames is not None and frame_index >= max_frames:
                break

            timestamp = frame_index / source.fps

            pose_frame = self.estimate_frame(
                image=frame,
                timestamp=timestamp,
                frame_index=frame_index,
            )

            if pose_frame is not None:
                sequence.add_frame(pose_frame)

            frame_index += 1
            pbar.update(1)

        pbar.close()

        logger.info(
            f"视频处理完成: {frame_index} 帧输入, "
            f"{sequence.num_frames} 帧有效姿态"
        )

        return sequence

    def close(self):
        """释放 MediaPipe 资源"""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def model_complexity(self) -> int:
        """当前模型复杂度"""
        return self._model_complexity

    @property
    def min_detection_confidence(self) -> float:
        """最小检测置信度"""
        return self._min_detection_confidence

    @property
    def min_tracking_confidence(self) -> float:
        """最小追踪置信度"""
        return self._min_tracking_confidence

    def __repr__(self) -> str:
        return (
            f"PoseEstimator(complexity={self._model_complexity}, "
            f"det_conf={self._min_detection_confidence}, "
            f"track_conf={self._min_tracking_confidence})"
        )
