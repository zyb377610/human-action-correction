"""
视频输入管理模块

提供统一的视频源抽象接口，支持摄像头实时流和视频文件两种输入。
所有视频源实现统一的 read/release 接口和上下文管理器协议。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoSource(ABC):
    """
    视频源抽象基类

    定义统一的视频读取接口，屏蔽摄像头和视频文件之间的差异。
    支持上下文管理器协议（with 语句）自动释放资源。

    使用示例:
        with CameraSource(0) as source:
            while True:
                success, frame = source.read()
                if not success:
                    break
    """

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧

        Returns:
            (success, frame) 元组：
            - success: 是否成功读取
            - frame: BGR 格式的 NumPy 数组，失败时为 None
        """
        pass

    @abstractmethod
    def release(self):
        """释放视频源资源"""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """视频帧率"""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """视频宽度（像素）"""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """视频高度（像素）"""
        pass

    @property
    def is_opened(self) -> bool:
        """视频源是否已打开"""
        return False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.release()
        return False


class CameraSource(VideoSource):
    """
    摄像头视频源

    从指定设备 ID 的摄像头读取实时视频流。

    Args:
        device_id: 摄像头设备 ID，默认为 0
        width: 请求的画面宽度（像素），None 表示使用默认值
        height: 请求的画面高度（像素），None 表示使用默认值
        target_fps: 请求的帧率，None 表示使用默认值
    """

    def __init__(
        self,
        device_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[float] = None,
    ):
        self._device_id = device_id
        self._cap = cv2.VideoCapture(device_id)

        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (device_id={device_id})")

        # 设置请求的分辨率和帧率
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if target_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧摄像头画面"""
        if not self._cap.isOpened():
            return False, None
        success, frame = self._cap.read()
        return success, frame if success else None

    def release(self):
        """释放摄像头资源"""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    @property
    def fps(self) -> float:
        """摄像头实际帧率"""
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        """画面宽度"""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """画面高度"""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def is_opened(self) -> bool:
        """摄像头是否已打开"""
        return self._cap is not None and self._cap.isOpened()

    def __repr__(self) -> str:
        return (f"CameraSource(device_id={self._device_id}, "
                f"{self.width}x{self.height}@{self.fps:.1f}fps)")


class FileSource(VideoSource):
    """
    视频文件源

    从视频文件读取帧数据。

    Args:
        file_path: 视频文件路径

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 无法打开视频文件
    """

    def __init__(self, file_path: str):
        self._path = Path(file_path)

        if not self._path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self._path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧视频画面"""
        if not self._cap.isOpened():
            return False, None
        success, frame = self._cap.read()
        return success, frame if success else None

    def release(self):
        """释放视频文件资源"""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    @property
    def fps(self) -> float:
        """视频文件帧率"""
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        """视频宽度"""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """视频高度"""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        """视频总帧数"""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> float:
        """视频时长（秒）"""
        fps = self.fps
        if fps <= 0:
            return 0.0
        return self.total_frames / fps

    @property
    def is_opened(self) -> bool:
        """视频文件是否已打开"""
        return self._cap is not None and self._cap.isOpened()

    def __repr__(self) -> str:
        return (f"FileSource(path='{self._path.name}', "
                f"{self.width}x{self.height}@{self.fps:.1f}fps, "
                f"{self.total_frames} frames)")
