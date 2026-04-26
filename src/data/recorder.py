"""
交互式标准动作录制器

提供倒计时→录制→预览→保存的完整工作流。
通过摄像头实时采集骨骼关键点，生成标准模板。
"""

import logging
import time
from enum import Enum, auto
from typing import List, Optional

import cv2

from src.pose_estimation.data_types import PoseFrame, PoseSequence
from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.video_source import CameraSource
from src.pose_estimation.visualizer import draw_skeleton
from src.data.template_library import TemplateLibrary
from src.data.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)


class RecorderState(Enum):
    """录制器状态"""
    IDLE = auto()        # 空闲 / 等待开始
    COUNTDOWN = auto()   # 倒计时中
    RECORDING = auto()   # 录制中
    PREVIEW = auto()     # 预览录制结果
    DONE = auto()        # 完成


class ActionRecorder:
    """
    交互式标准动作录制器

    工作流:
        1. 打开摄像头，显示实时画面
        2. 按 SPACE 开始录制 → 3 秒倒计时
        3. 倒计时结束后自动开始录制
        4. 按 SPACE 停止录制 → 进入预览
        5. 预览模式: 按 S 保存 / 按 R 重录 / 按 Q 退出

    使用示例:
        recorder = ActionRecorder()
        recorder.record("squat", "standard_01")
    """

    def __init__(
        self,
        camera_id: int = 0,
        countdown_seconds: int = 3,
        max_record_seconds: float = 30.0,
        preprocess: bool = True,
        target_frames: int = 60,
        templates_dir: Optional[str] = None,
    ):
        """
        Args:
            camera_id: 摄像头设备 ID
            countdown_seconds: 倒计时秒数
            max_record_seconds: 最大录制时长（秒），防止忘记停止
            preprocess: 保存前是否执行预处理
            target_frames: 归一化帧数
            templates_dir: 模板库目录
        """
        self._camera_id = camera_id
        self._countdown = countdown_seconds
        self._max_seconds = max_record_seconds
        self._preprocess = preprocess
        self._target_frames = target_frames

        self._library = TemplateLibrary(templates_dir)
        self._estimator = PoseEstimator()

        self._state = RecorderState.IDLE
        self._frames: List[PoseFrame] = []

    @property
    def state(self) -> RecorderState:
        return self._state

    def record(self, action: str, template_name: str) -> Optional[PoseSequence]:
        """
        启动交互式录制

        Args:
            action: 动作类别名称
            template_name: 模板名称

        Returns:
            保存成功则返回 PoseSequence，否则 None
        """
        camera = CameraSource(self._camera_id)
        camera.open()
        fps = camera.fps or 30.0

        window_name = f"录制: {action} / {template_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self._state = RecorderState.IDLE
        self._frames = []
        countdown_start = 0.0
        record_start = 0.0
        result_sequence: Optional[PoseSequence] = None

        try:
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    break

                # 姿态估计
                timestamp = time.time()
                frame_index = len(self._frames)
                pose_frame = self._estimator.estimate_frame(
                    frame, timestamp, frame_index
                )

                # 绘制骨骼
                display = frame.copy()
                if pose_frame:
                    display = draw_skeleton(display, pose_frame)

                # 状态机
                if self._state == RecorderState.IDLE:
                    self._draw_text(display, "按 SPACE 开始录制 | Q 退出", (20, 40))

                elif self._state == RecorderState.COUNTDOWN:
                    elapsed = time.time() - countdown_start
                    remain = self._countdown - int(elapsed)
                    if remain > 0:
                        self._draw_text(
                            display, f"准备... {remain}", (200, 240),
                            font_scale=3.0, color=(0, 0, 255), thickness=4
                        )
                    else:
                        self._state = RecorderState.RECORDING
                        record_start = time.time()
                        self._frames = []
                        logger.info("开始录制")

                elif self._state == RecorderState.RECORDING:
                    if pose_frame:
                        self._frames.append(pose_frame)
                    elapsed = time.time() - record_start
                    self._draw_text(
                        display,
                        f"● 录制中 {elapsed:.1f}s | SPACE 停止",
                        (20, 40), color=(0, 0, 255)
                    )
                    # 超时自动停止
                    if elapsed >= self._max_seconds:
                        self._state = RecorderState.PREVIEW
                        logger.info(f"录制超时自动停止 ({self._max_seconds}s)")

                elif self._state == RecorderState.PREVIEW:
                    self._draw_text(
                        display,
                        f"录制完成: {len(self._frames)} 帧 | S:保存 R:重录 Q:退出",
                        (20, 40), color=(0, 255, 0)
                    )

                cv2.imshow(window_name, display)

                # 键盘处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    break

                elif key == ord(' '):  # SPACE
                    if self._state == RecorderState.IDLE:
                        self._state = RecorderState.COUNTDOWN
                        countdown_start = time.time()
                        logger.info(f"倒计时 {self._countdown} 秒")
                    elif self._state == RecorderState.RECORDING:
                        self._state = RecorderState.PREVIEW
                        logger.info(f"停止录制，共 {len(self._frames)} 帧")

                elif key == ord('s') or key == ord('S'):
                    if self._state == RecorderState.PREVIEW and self._frames:
                        sequence = PoseSequence(self._frames, fps)
                        if self._preprocess:
                            sequence = preprocess_pipeline(
                                sequence, target_frames=self._target_frames
                            )
                        self._library.add_template(action, sequence, template_name)
                        result_sequence = sequence
                        self._state = RecorderState.DONE
                        logger.info(
                            f"模板已保存: {action}/{template_name} "
                            f"({sequence.num_frames} 帧)"
                        )
                        break

                elif key == ord('r') or key == ord('R'):
                    if self._state == RecorderState.PREVIEW:
                        self._state = RecorderState.IDLE
                        self._frames = []
                        logger.info("重新录制")

        finally:
            camera.release()
            cv2.destroyWindow(window_name)

        return result_sequence

    @staticmethod
    def _draw_text(
        image,
        text: str,
        position: tuple,
        font_scale: float = 0.8,
        color: tuple = (255, 255, 255),
        thickness: int = 2,
    ):
        """在图像上绘制文本（带黑色背景增强可读性）"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(image, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
