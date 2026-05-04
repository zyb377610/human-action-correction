"""
姿态估计核心模块

封装 MediaPipe PoseLandmarker (Tasks API)，提供统一的姿态估计接口。
支持单帧图像和视频流两种输入模式，输出标准化的 PoseFrame/PoseSequence。

增强：
- PoseSmoother：按可见度加权的 EMA 平滑，降低摄像头实时抖动
- StreamingPoseEstimator 辅助：内部复用 PoseEstimator，并在丢检时保持上一帧骨骼
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


class PoseSmoother:
    """
    关键点轻量 EMA 平滑器（实时模式专用）

    在 MediaPipe LIVE_STREAM 的基础上做一层"锦上添花"的微平滑：
        smoothed[j] = a * raw[j] + (1 - a) * prev[j]
        其中 a = alpha * max(visibility, min_alpha_floor)

    默认 alpha=0.75 更轻，能跟上运动；丢失时仅短暂保持（3帧）。
    """

    def __init__(
        self,
        alpha: float = 0.75,
        min_alpha_floor: float = 0.5,
        max_hold_frames: int = 3,
    ):
        """
        Args:
            alpha: 基准平滑系数，越大越紧跟当前帧（实时建议 0.7-0.85）
            min_alpha_floor: visibility 很低时的 alpha 下限
            max_hold_frames: 丢检时保留上次骨骼的最大帧数（3 帧≈100ms@30fps）
        """
        self._alpha = alpha
        self._floor = min_alpha_floor
        self._prev: Optional[np.ndarray] = None
        self._max_hold = max_hold_frames
        self._miss_count = 0

    @property
    def last(self) -> Optional[np.ndarray]:
        """上一次有效的平滑后骨骼 (33, 4)"""
        return self._prev

    def reset(self):
        """重置状态（切换用户/会话时调用）"""
        self._prev = None
        self._miss_count = 0

    def update(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        更新平滑状态，返回平滑后的骨骼。

        Args:
            landmarks: 当前帧原始关键点 (33, 4)，None 表示本帧丢检

        Returns:
            平滑后关键点 (33, 4)，若尚无历史且丢检则为 None
        """
        if landmarks is None:
            # 丢检：若有历史且未超过 hold 帧数，返回上一次
            if self._prev is not None and self._miss_count < self._max_hold:
                self._miss_count += 1
                return self._prev
            # 超过 hold 阈值 → 彻底丢失
            self._miss_count += 1
            return None

        # 首帧，直接接受
        if self._prev is None:
            self._prev = landmarks.astype(np.float64).copy()
            self._miss_count = 0
            return self._prev.copy()

        # 正常帧：按 visibility 加权 EMA
        self._miss_count = 0
        prev = self._prev
        cur = landmarks.astype(np.float64)

        vis = np.clip(cur[:, 3:4], 0.0, 1.0)       # (33, 1)
        # alpha_j = alpha * max(vis_j, floor)
        a = self._alpha * np.maximum(vis, self._floor)
        # 只对 xyz 平滑，visibility 直接用当前帧值
        xyz = a * cur[:, :3] + (1.0 - a) * prev[:, :3]
        smoothed = np.concatenate([xyz, cur[:, 3:4]], axis=1)

        self._prev = smoothed
        return smoothed.copy()


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


class StreamingPoseEstimator:
    """
    实时流式姿态估计器（LIVE_STREAM 模式）

    相比 PoseEstimator（IMAGE 模式）：
    - 使用 MediaPipe 内置的跨帧追踪，骨骼更稳定、对运动跟得更紧
    - 丢检时会自动依赖上一帧进行追踪（官方时序滤波）
    - 输入 timestamp_ms 必须单调递增，内部回调异步返回结果

    用法:
        streamer = StreamingPoseEstimator(model_complexity=1)
        # 同步调用
        landmarks = streamer.process(frame_bgr, timestamp_ms)
        # landmarks: (33, 4) 或 None
    """

    def __init__(
        self,
        model_complexity: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        config = get_config(config_path)
        pose_cfg = config.get_section("pose_estimation")

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

        if model_path is not None:
            self._model_path = Path(model_path)
        else:
            model_file = _MODEL_FILES.get(self._model_complexity,
                                          _MODEL_FILES[1])
            self._model_path = _DEFAULT_MODEL_DIR / model_file

        if not self._model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self._model_path}")

        self._latest_result = None  # 异步回调写入
        self._landmarker = None
        self._last_ts_ms: int = -1
        self._init_landmarker()

        logger.info(
            f"StreamingPoseEstimator 初始化: complexity={self._model_complexity}"
        )

    def _on_result(self, result, output_image, timestamp_ms):
        """MediaPipe LIVE_STREAM 异步回调：保存最近结果"""
        self._latest_result = (result, timestamp_ms)

    def _init_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        with open(str(self._model_path), "rb") as f:
            model_data = f.read()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_pose_presence_confidence=self._min_tracking_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            num_poses=1,
            result_callback=self._on_result,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def process(
        self,
        image_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        处理一帧并同步返回关键点数组 (33, 4)。

        若同一 timestamp_ms 之前已送入会导致 MediaPipe 报错，
        因此这里强制保证时间戳单调递增。

        Args:
            image_bgr: BGR 图像
            timestamp_ms: 毫秒时间戳；None 时使用 time.time()*1000

        Returns:
            landmarks (33, 4) [x,y,z,visibility]；本帧未检测到人体时 None
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # 强制单调递增
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 清空上次结果并送入
        self._latest_result = None
        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception as e:
            logger.warning(f"LIVE_STREAM detect_async 异常，回退: {e}")
            return None

        # 等待回调（最长 80ms，足以覆盖 Full 模型 GPU/CPU 推理）
        deadline = time.time() + 0.08
        while self._latest_result is None and time.time() < deadline:
            time.sleep(0.001)

        if self._latest_result is None:
            # 本帧未及时返回；下一帧会重新送入，UI 短暂保持骨骼
            return None

        result, _ts = self._latest_result
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        pose_landmarks = result.pose_landmarks[0]
        if len(pose_landmarks) < NUM_LANDMARKS:
            return None

        arr = np.zeros((NUM_LANDMARKS, 4), dtype=np.float64)
        for i, lm in enumerate(pose_landmarks):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
            arr[i, 3] = lm.visibility if lm.visibility is not None else 0.0
        return arr

    def reset(self):
        """重置内部状态（切换会话时调用）"""
        self._last_ts_ms = -1
        self._latest_result = None

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
