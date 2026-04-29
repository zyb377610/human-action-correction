"""
会话状态管理

SessionManager 管理摄像头录制状态、帧缓存、分析结果缓存和模板列表。
配合 Gradio 的 gr.State 使用。

增强功能：
- 实时模式状态（COUNTDOWN / REALTIME）
- 实时反馈缓存
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .data_types import AnalysisResult

logger = logging.getLogger(__name__)


class RecordingState(str, Enum):
    """录制/实时状态枚举"""
    IDLE = "idle"              # 空闲
    RECORDING = "recording"    # 录制中（视频分析 Tab 的摄像头录制）
    ANALYZING = "analyzing"    # 分析中
    COUNTDOWN = "countdown"    # 倒计时准备（实时模式）
    REALTIME = "realtime"      # 实时分析中（实时模式）


class SessionManager:
    """
    会话状态管理器

    管理内容：
    1. 摄像头录制状态（idle / recording / analyzing / countdown / realtime）
    2. 帧缓存（录制/实时期间的 landmarks 列表）
    3. 最近分析结果缓存
    4. 模板列表（动作类型下拉框选项）
    5. 实时反馈缓存（最新的 FeedbackSnapshot）
    """

    def __init__(self, template_library=None):
        """
        Args:
            template_library: TemplateLibrary 实例
        """
        self._state = RecordingState.IDLE
        self._frame_buffer: List[np.ndarray] = []  # 每帧 landmarks (33,4)
        self._last_result: Optional[AnalysisResult] = None
        self._library = template_library
        self._action_list: List[str] = []

        # 实时模式相关
        self._countdown_start: float = 0.0  # 倒计时开始时间
        self._countdown_duration: float = 3.0  # 倒计时时长（秒）
        self._realtime_frame_index: int = 0  # 实时帧计数器
        self._last_feedback = None  # 最新的 FeedbackSnapshot

        # 初始加载模板列表
        self.refresh_action_list()

    # ===== 录制状态管理 =====

    @property
    def state(self) -> RecordingState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == RecordingState.IDLE

    @property
    def is_recording(self) -> bool:
        return self._state == RecordingState.RECORDING

    @property
    def is_analyzing(self) -> bool:
        return self._state == RecordingState.ANALYZING

    @property
    def is_countdown(self) -> bool:
        return self._state == RecordingState.COUNTDOWN

    @property
    def is_realtime(self) -> bool:
        return self._state == RecordingState.REALTIME

    # ----- 视频分析 Tab 的录制 -----

    def start_recording(self):
        """开始录制（视频分析 Tab 摄像头模式）"""
        self._state = RecordingState.RECORDING
        self._frame_buffer.clear()
        logger.info("开始录制")

    def add_frame(self, landmarks: np.ndarray):
        """
        添加一帧 landmarks 到缓存

        Args:
            landmarks: (33, 4) 关键点数组
        """
        if self._state in (RecordingState.RECORDING, RecordingState.REALTIME):
            self._frame_buffer.append(landmarks.copy())

    def stop_recording(self) -> Optional[np.ndarray]:
        """
        停止录制，返回缓存的骨骼序列

        Returns:
            numpy 数组 (T, 33, 4)，无数据时返回 None
        """
        self._state = RecordingState.ANALYZING
        logger.info(f"停止录制，缓存帧数: {len(self._frame_buffer)}")

        if len(self._frame_buffer) < 5:
            self._state = RecordingState.IDLE
            return None

        sequence = np.stack(self._frame_buffer, axis=0)  # (T, 33, 4)
        return sequence

    def finish_analysis(self, result: Optional[AnalysisResult] = None):
        """分析完成，重置状态"""
        self._state = RecordingState.IDLE
        if result is not None:
            self._last_result = result
        self._frame_buffer.clear()
        logger.info("分析完成，状态重置")

    # ----- 实时模式 -----

    def start_countdown(self):
        """开始倒计时（实时模式）"""
        self._state = RecordingState.COUNTDOWN
        self._countdown_start = time.time()
        self._frame_buffer.clear()
        self._realtime_frame_index = 0
        self._last_feedback = None
        logger.info("实时模式：倒计时开始")

    @property
    def countdown_remaining(self) -> float:
        """倒计时剩余秒数"""
        if self._state != RecordingState.COUNTDOWN:
            return 0.0
        elapsed = time.time() - self._countdown_start
        remaining = self._countdown_duration - elapsed
        return max(0.0, remaining)

    @property
    def countdown_finished(self) -> bool:
        """倒计时是否结束"""
        return self.countdown_remaining <= 0.0

    def start_realtime(self):
        """倒计时结束后进入实时分析"""
        self._state = RecordingState.REALTIME
        self._frame_buffer.clear()
        self._realtime_frame_index = 0
        logger.info("实时模式：开始实时分析")

    def advance_realtime_frame(self) -> int:
        """
        推进实时帧计数器

        Returns:
            当前帧索引
        """
        idx = self._realtime_frame_index
        self._realtime_frame_index += 1
        return idx

    @property
    def realtime_frame_index(self) -> int:
        return self._realtime_frame_index

    def stop_realtime(self) -> Optional[np.ndarray]:
        """
        结束实时分析，返回完整帧序列

        Returns:
            numpy 数组 (T, 33, 4)，无数据时返回 None
        """
        self._state = RecordingState.ANALYZING
        logger.info(
            f"实时模式结束，缓存帧数: {len(self._frame_buffer)}"
        )

        if len(self._frame_buffer) < 5:
            self._state = RecordingState.IDLE
            return None

        sequence = np.stack(self._frame_buffer, axis=0)
        return sequence

    # ----- 实时反馈缓存 -----

    @property
    def last_feedback(self):
        """最新的 FeedbackSnapshot"""
        return self._last_feedback

    @last_feedback.setter
    def last_feedback(self, value):
        self._last_feedback = value

    # ===== 通用方法 =====

    def reset(self):
        """完全重置"""
        self._state = RecordingState.IDLE
        self._frame_buffer.clear()
        self._last_result = None
        self._last_feedback = None
        self._realtime_frame_index = 0

    # ===== 结果缓存 =====

    @property
    def last_result(self) -> Optional[AnalysisResult]:
        return self._last_result

    @last_result.setter
    def last_result(self, value: AnalysisResult):
        self._last_result = value

    @property
    def frame_count(self) -> int:
        return len(self._frame_buffer)

    # ===== 模板列表管理 =====

    def refresh_action_list(self):
        """刷新可用动作类型列表"""
        if self._library is not None:
            self._action_list = self._library.list_actions()
        else:
            self._action_list = []
        logger.info(f"动作列表刷新: {self._action_list}")

    def get_action_list(self) -> List[str]:
        """获取可用动作类型列表"""
        return list(self._action_list)

    def get_action_choices(self) -> List[str]:
        """
        获取下拉框选项（含"自动识别"）

        Returns:
            ["自动识别", "squat", "arm_raise", ...]
        """
        choices = ["自动识别"]
        choices.extend(self._action_list)
        return choices

    @property
    def state_display(self) -> str:
        """获取状态的中文显示"""
        mapping = {
            RecordingState.IDLE: "⏸️ 空闲",
            RecordingState.RECORDING: "🔴 录制中",
            RecordingState.ANALYZING: "⏳ 分析中…",
            RecordingState.COUNTDOWN: "⏱️ 倒计时准备中…",
            RecordingState.REALTIME: "🟢 实时分析中",
        }
        return mapping.get(self._state, str(self._state))
