"""
应用层端到端流水线

AppPipeline 是 UI 层与所有算法模块之间的唯一桥梁。
串联：姿态估计 → 分类 → DTW 对比 → 矫正反馈 → 可视化。
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.data_types import PoseFrame, PoseSequence, PoseLandmark
from src.pose_estimation.visualizer import draw_skeleton
from src.pose_estimation.video_source import FileSource
from src.data.template_library import TemplateLibrary
from src.correction.pipeline import CorrectionPipeline
from src.correction.report_visualizer import ReportVisualizer

from .data_types import AnalysisResult, ProcessedFrame

logger = logging.getLogger(__name__)

# 动作中文名称映射
ACTION_DISPLAY_NAMES = {
    "squat": "深蹲",
    "arm_raise": "手臂举起",
    "side_bend": "侧弯",
    "lunge": "弓步",
    "standing_stretch": "站立拉伸",
}

# 最大处理帧数
MAX_FRAMES = 300


class AppPipeline:
    """
    应用层端到端流水线

    提供 4 个核心方法供 UI 回调使用：
    - analyze_video(): 视频文件 → 完整分析结果
    - analyze_sequence(): 骨骼序列 → 完整分析结果
    - process_camera_frame(): 单帧 → 骨骼叠加图
    - record_template(): 视频 → 存入模板库
    """

    def __init__(
        self,
        templates_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_type: str = "bilstm",
        output_dir: str = "outputs/app",
    ):
        """
        Args:
            templates_dir: 模板库目录
            checkpoint_path: 分类模型 checkpoint（None 仅支持指定模式）
            model_type: 分类模型类型
            output_dir: 输出文件目录（图表、视频等）
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 姿态估计器
        self._estimator = PoseEstimator()

        # 模板库
        self._library = TemplateLibrary(templates_dir)

        # 矫正流水线
        self._correction_pipeline = CorrectionPipeline(
            templates_dir=templates_dir,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
        )

        # 报告可视化
        self._visualizer = ReportVisualizer()

        logger.info("AppPipeline 初始化完成")

    # ================================================================
    # 核心方法 1: 视频文件分析
    # ================================================================

    def analyze_video(
        self,
        video_path: str,
        action_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> AnalysisResult:
        """
        对视频文件执行端到端矫正分析

        Args:
            video_path: 视频文件路径
            action_name: 动作类别（None 则自动识别）
            progress_callback: 进度回调 fn(step, total, message)

        Returns:
            AnalysisResult
        """
        total_steps = 4

        def _progress(step: int, msg: str):
            if progress_callback:
                progress_callback(step, total_steps, msg)

        # Step 1: 骨骼提取
        _progress(1, "正在提取骨骼关键点…")
        with FileSource(video_path) as source:
            sequence = self._estimator.estimate_video(
                source, max_frames=MAX_FRAMES, show_progress=False
            )

        if sequence.num_frames < 5:
            return AnalysisResult(
                report_text="⚠️ 视频中未检测到足够的人体姿态帧（最少需要 5 帧）。\n"
                            "请确保视频中有清晰的全身画面。",
            )

        # Step 2-4: 矫正分析
        return self._analyze_sequence_internal(
            sequence, action_name, progress_callback, step_offset=1
        )

    # ================================================================
    # 核心方法 2: 骨骼序列直接分析
    # ================================================================

    def analyze_sequence(
        self,
        sequence_data: np.ndarray,
        action_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> AnalysisResult:
        """
        对已有骨骼序列执行矫正分析（跳过姿态估计）

        Args:
            sequence_data: numpy 数组 (T, 33, 4) — [x, y, z, visibility]
            action_name: 动作类别
            progress_callback: 进度回调

        Returns:
            AnalysisResult
        """
        # 将 numpy 数组转为 PoseSequence
        sequence = self._array_to_sequence(sequence_data)

        return self._analyze_sequence_internal(
            sequence, action_name, progress_callback, step_offset=0
        )

    # ================================================================
    # 核心方法 3: 摄像头单帧处理
    # ================================================================

    def process_camera_frame(self, frame: np.ndarray) -> ProcessedFrame:
        """
        处理摄像头单帧，返回带骨骼叠加的图像

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            ProcessedFrame
        """
        if frame is None or frame.size == 0:
            return ProcessedFrame(annotated_image=frame)

        pose_frame = self._estimator.estimate_frame(frame)

        if pose_frame is None:
            return ProcessedFrame(annotated_image=frame.copy())

        # 绘制骨骼
        annotated = frame.copy()
        draw_skeleton(annotated, pose_frame)

        # 提取 landmarks 为 numpy 数组
        landmarks = pose_frame.to_numpy()  # (33, 4)

        return ProcessedFrame(
            annotated_image=annotated,
            landmarks=landmarks,
        )

    # ================================================================
    # 核心方法 4: 模板录入
    # ================================================================

    def record_template(
        self,
        video_path: str,
        action_name: str,
        template_name: Optional[str] = None,
    ) -> bool:
        """
        从视频提取骨骼序列并保存为标准模板

        Args:
            video_path: 视频文件路径
            action_name: 动作类别名称
            template_name: 模板名称（None 则自动生成）

        Returns:
            是否成功
        """
        try:
            with FileSource(video_path) as source:
                sequence = self._estimator.estimate_video(
                    source, max_frames=MAX_FRAMES, show_progress=False
                )

            if sequence.num_frames < 5:
                logger.warning(f"视频帧数不足: {sequence.num_frames}")
                return False

            if template_name is None:
                template_name = f"template_{int(time.time())}"

            # 确保动作类别存在
            display_name = ACTION_DISPLAY_NAMES.get(action_name, action_name)
            if action_name not in self._library.list_actions():
                self._library.add_action(action_name, display_name)

            self._library.add_template(action_name, sequence, template_name)

            logger.info(
                f"模板录入成功: {action_name}/{template_name} "
                f"({sequence.num_frames} 帧)"
            )
            return True

        except Exception as e:
            logger.error(f"模板录入失败: {e}")
            return False

    # ================================================================
    # 辅助属性
    # ================================================================

    @property
    def template_library(self) -> TemplateLibrary:
        """获取模板库实例"""
        return self._library

    def get_action_list(self):
        """获取可用动作类别列表"""
        return self._library.list_actions()

    def get_template_info(self):
        """获取模板库信息，返回列表 [{action, display_name, count, templates}]"""
        info_list = []
        for action in self._library.list_actions():
            action_info = self._library.get_action_info(action) or {}
            templates = self._library.list_templates(action)
            info_list.append({
                "action": action,
                "display_name": action_info.get("display_name", action),
                "count": len(templates),
                "templates": templates,
            })
        return info_list

    # ================================================================
    # 内部方法
    # ================================================================

    def _analyze_sequence_internal(
        self,
        sequence: PoseSequence,
        action_name: Optional[str],
        progress_callback: Optional[Callable],
        step_offset: int = 0,
    ) -> AnalysisResult:
        """
        内部分析方法（从 PoseSequence 开始）

        Args:
            sequence: 骨骼序列
            action_name: 动作类别
            progress_callback: 进度回调
            step_offset: 步骤偏移量（视频模式偏移 1）
        """
        total_steps = 4

        def _progress(step: int, msg: str):
            if progress_callback:
                progress_callback(step_offset + step, total_steps, msg)

        try:
            # Step 2: 分类
            _progress(2, "正在分析动作类型…")

            # Step 3: DTW 对比
            _progress(3, "正在对比标准动作…")

            # Step 4: 生成报告
            _progress(4, "正在生成矫正报告…")

            # 调用矫正流水线（内部完成分类+对比+偏差+反馈）
            report = self._correction_pipeline.analyze(
                user_sequence=sequence,
                action_name=action_name,
            )

            # 生成可视化文件
            deviation_plot_path = None
            try:
                plot_path = self._output_dir / f"deviation_{int(time.time())}.png"
                self._visualizer.plot_deviation_bar(report, save_path=str(plot_path))
                deviation_plot_path = str(plot_path)
            except Exception as e:
                logger.warning(f"生成偏差图失败: {e}")

            # 构建结果
            display_name = ACTION_DISPLAY_NAMES.get(
                report.action_name, report.action_display_name or report.action_name
            )

            result = AnalysisResult(
                action_name=report.action_name,
                action_display_name=display_name,
                quality_score=report.quality_score,
                similarity=report.similarity,
                report_text=report.to_text(),
                deviation_plot_path=deviation_plot_path,
                skeleton_video_path=None,  # 暂不生成骨骼视频
                corrections=list(report.corrections),
                report=report,
                confidence=report.confidence,
            )

            return result

        except ValueError as e:
            err_msg = str(e)
            return AnalysisResult(
                report_text=(
                    f"⚠️ 分析失败: {err_msg}\n\n"
                    "可能原因：\n"
                    "1. 模板库中没有对应动作的标准模板\n"
                    '2. 请先在"模板管理"中录入标准动作模板'
                ),
            )
        except Exception as e:
            err_msg = str(e)
            logger.error(f"分析异常: {err_msg}", exc_info=True)
            return AnalysisResult(
                report_text=f"❌ 分析过程出现错误: {err_msg}",
            )

    @staticmethod
    def _array_to_sequence(data: np.ndarray, fps: float = 30.0) -> PoseSequence:
        """
        将 numpy 数组 (T, 33, 4) 转为 PoseSequence

        Args:
            data: 关键点数组
            fps: 帧率

        Returns:
            PoseSequence
        """
        sequence = PoseSequence(fps=fps)
        for i in range(data.shape[0]):
            landmarks = []
            for j in range(data.shape[1]):
                landmarks.append(PoseLandmark(
                    x=float(data[i, j, 0]),
                    y=float(data[i, j, 1]),
                    z=float(data[i, j, 2]),
                    visibility=float(data[i, j, 3]) if data.shape[2] > 3 else 1.0,
                ))
            frame = PoseFrame(
                timestamp=i / fps,
                frame_index=i,
                landmarks=landmarks,
            )
            sequence.add_frame(frame)
        return sequence

    def close(self):
        """释放资源"""
        if hasattr(self, '_estimator') and self._estimator:
            self._estimator.close()
