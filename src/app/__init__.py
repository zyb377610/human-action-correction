"""
应用入口模块

提供 Web 交互界面和端到端流水线。
"""

from .data_types import AnalysisResult, ProcessedFrame
from .pipeline import AppPipeline
from .session import SessionManager, RecordingState
from .gradio_ui import create_gradio_app

__all__ = [
    "AppPipeline",
    "SessionManager",
    "RecordingState",
    "AnalysisResult",
    "ProcessedFrame",
    "create_gradio_app",
]