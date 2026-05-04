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

from src.pose_estimation.estimator import (
    PoseEstimator,
    PoseSmoother,
    StreamingPoseEstimator,
)
from src.pose_estimation.data_types import PoseFrame, PoseSequence, PoseLandmark
from src.pose_estimation.visualizer import draw_skeleton
from src.pose_estimation.video_source import FileSource
from src.data.template_library import TemplateLibrary
from src.correction.pipeline import CorrectionPipeline
from src.correction.report_visualizer import ReportVisualizer
from src.correction.realtime_feedback import (
    RealtimeFeedbackEngine,
    FeedbackSnapshot,
)
from src.action_comparison.comparison_video import generate_comparison_video

from .data_types import AnalysisResult, ProcessedFrame, TemplateRecordResult

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

# DTW 算法显示名称映射
ALGORITHM_DISPLAY_NAMES = {
    "dtw": "经典 DTW",
    "fastdtw": "FastDTW",
    "ddtw": "DerivativeDTW",
    "auto": "自动选择",
}

# UI 下拉框算法选项
ALGORITHM_CHOICES = [
    ("经典 DTW", "dtw"),
    ("FastDTW", "fastdtw"),
    ("DerivativeDTW", "ddtw"),
    ("自动选择（推荐）", "auto"),
]


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

        # 姿态估计器（离线：IMAGE 模式）
        self._estimator = PoseEstimator()

        # 实时链路专用流式估计器（LIVE_STREAM 模式，支持时序追踪）
        # 摄像头和实时反馈分别用一个独立实例，避免时间戳冲突
        self._cam_streamer: Optional[StreamingPoseEstimator] = None
        self._rt_streamer: Optional[StreamingPoseEstimator] = None

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

        # 当前 DTW 算法设置 (dtw / fastdtw / ddtw / auto)
        self._algorithm = "auto"

        # 实时反馈引擎（在 init_realtime_session 时创建）
        self._realtime_engine: Optional[RealtimeFeedbackEngine] = None

        # 实时链路轻量 EMA 平滑器（MediaPipe 时序追踪之上的微平滑）
        # alpha 调高（0.80）、hold 调小（3 帧）→ 跟动作更紧
        self._cam_smoother = PoseSmoother(alpha=0.80, min_alpha_floor=0.55,
                                           max_hold_frames=3)
        self._rt_smoother = PoseSmoother(alpha=0.80, min_alpha_floor=0.55,
                                          max_hold_frames=3)

        logger.info("AppPipeline 初始化完成")

    def _ensure_cam_streamer(self) -> StreamingPoseEstimator:
        """摄像头流式估计器按需创建"""
        if self._cam_streamer is None:
            self._cam_streamer = StreamingPoseEstimator()
        return self._cam_streamer

    def _ensure_rt_streamer(self) -> StreamingPoseEstimator:
        """实时反馈流式估计器按需创建"""
        if self._rt_streamer is None:
            self._rt_streamer = StreamingPoseEstimator()
        return self._rt_streamer

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
        total_steps = 5  # 骨骼提取 + 分类 + DTW + 报告 + 对比视频

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
            sequence, action_name, progress_callback, step_offset=1,
            video_path=video_path,
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

        使用 MediaPipe LIVE_STREAM 模式 + 轻量 EMA 微平滑，
        既有官方时序追踪的稳定性，又保留 EMA 的抗抖动能力。

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            ProcessedFrame
        """
        if frame is None or frame.size == 0:
            return ProcessedFrame(annotated_image=frame)

        streamer = self._ensure_cam_streamer()
        raw = streamer.process(frame)
        smoothed = self._cam_smoother.update(raw)

        if smoothed is None:
            # 既无当前检测，也无历史可复用
            return ProcessedFrame(annotated_image=frame.copy())

        # 用平滑后的 landmarks 构造 PoseFrame 用于绘制
        draw_frame = self._landmarks_to_poseframe(smoothed)
        annotated = frame.copy()
        draw_skeleton(annotated, draw_frame)

        return ProcessedFrame(
            annotated_image=annotated,
            landmarks=smoothed,
        )

    @staticmethod
    def _landmarks_to_poseframe(landmarks: np.ndarray,
                                 frame_index: int = 0,
                                 timestamp: float = 0.0) -> PoseFrame:
        """将 (33, 4) 关键点数组包装为 PoseFrame（用于可视化）"""
        lm_list = [
            PoseLandmark(
                x=float(landmarks[j, 0]),
                y=float(landmarks[j, 1]),
                z=float(landmarks[j, 2]),
                visibility=float(landmarks[j, 3]) if landmarks.shape[1] > 3 else 1.0,
            )
            for j in range(landmarks.shape[0])
        ]
        return PoseFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            landmarks=lm_list,
        )

    def reset_camera_smoothers(self):
        """在会话切换时重置摄像头/实时链路的平滑器与流式估计器"""
        self._cam_smoother.reset()
        self._rt_smoother.reset()
        if self._cam_streamer is not None:
            self._cam_streamer.reset()
        if self._rt_streamer is not None:
            self._rt_streamer.reset()

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

            # 模板入库后立即生成骨骼演示视频，避免后续展示时延迟
            try:
                self._build_template_demo(action_name, template_name, sequence)
            except Exception as _e:
                logger.warning(f"预生成模板演示视频失败（不影响录入）: {_e}")

            logger.info(
                f"模板录入成功: {action_name}/{template_name} "
                f"({sequence.num_frames} 帧)"
            )
            return True

        except Exception as e:
            logger.error(f"模板录入失败: {e}")
            return False

    def record_template_with_error(
        self,
        video_path: str,
        action_name: str,
        template_name: Optional[str] = None,
    ) -> TemplateRecordResult:
        """
        从视频提取骨骼序列并保存为标准模板（带详细错误信息）

        与 record_template 功能相同，但返回详细的成功/失败结果。

        Args:
            video_path: 视频文件路径
            action_name: 动作类别名称
            template_name: 模板名称（None 则自动生成）

        Returns:
            TemplateRecordResult（含 success 和 error 字段）
        """
        from pathlib import Path as _Path
        video_file = _Path(video_path)

        # 检查文件存在
        if not video_file.exists():
            return TemplateRecordResult(
                success=False, error=f"视频文件不存在: {video_path}"
            )

        # 检查文件扩展名
        suffix = video_file.suffix.lower()
        supported = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".wmv", ".m4v"}
        if suffix not in supported:
            logger.warning(f"视频格式可能不被支持: {suffix}，尝试用 OpenCV 打开…")

        # 检查 OpenCV 能否打开
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            cap.release()
            return TemplateRecordResult(
                success=False,
                error=(
                    f"无法打开视频文件（格式或编码不支持）: {video_file.name}\n\n"
                    "请尝试以下方法：\n"
                    "1. 用格式工厂 / HandBrake 将视频转为 MP4 (H.264) 格式\n"
                    "2. 使用「摄像头录制」方式重新录制标准动作"
                ),
            )
        # 读取一帧验证
        ok, _ = cap.read()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if not ok:
            return TemplateRecordResult(
                success=False,
                error=(
                    f"视频文件可打开但无法读取帧数据: {video_file.name}\n"
                    "该视频可能已损坏或使用了不支持的编码格式，请转换为 MP4 后重试。"
                ),
            )

        # 使用标准方法录入
        try:
            success = self.record_template(video_path, action_name, template_name)
            if success:
                return TemplateRecordResult(success=True)
            else:
                return TemplateRecordResult(
                    success=False,
                    error=(
                        f"视频处理失败 ({total_frames} 帧, {fps:.1f} fps)。\n"
                        "可能原因：动作识别帧数不足（需 ≥5 帧有姿态），或视频中未检测到人体。"
                    ),
                )
        except Exception as e:
            logger.error(f"模板录入异常: {e}", exc_info=True)
            return TemplateRecordResult(
                success=False, error=f"处理异常: {str(e)}"
            )

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

    def delete_template(self, action: str, template_name: str,
                        auto_remove_empty_action: bool = True) -> dict:
        """
        删除指定动作下的模板。当该动作下最后一个模板被删除时，
        可选地同时移除该动作类别，避免其在下拉列表中继续占位。

        Returns:
            {"success": bool, "message": str, "action_removed": bool}
        """
        try:
            templates_before = self._library.list_templates(action)
            if template_name not in templates_before:
                return {
                    "success": False,
                    "message": f"模板不存在: {action}/{template_name}",
                    "action_removed": False,
                }
            self._library.delete_template(action, template_name)
            templates_after = self._library.list_templates(action)
            action_removed = False
            if auto_remove_empty_action and len(templates_after) == 0:
                action_removed = self._library.delete_action(action)
            msg = f"✅ 已删除模板: {action}/{template_name}"
            if action_removed:
                msg += f"（该动作已无模板，动作类别 '{action}' 也已一并移除）"
            return {
                "success": True,
                "message": msg,
                "action_removed": action_removed,
            }
        except Exception as e:
            logger.error(f"删除模板失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"❌ 删除失败: {e}",
                "action_removed": False,
            }

    def delete_action(self, action: str) -> dict:
        """删除整个动作类别及其所有模板"""
        try:
            ok = self._library.delete_action(action)
            if ok:
                return {
                    "success": True,
                    "message": f"✅ 已删除动作类别: {action}",
                }
            return {
                "success": False,
                "message": f"⚠️ 动作类别不存在: {action}",
            }
        except Exception as e:
            logger.error(f"删除动作失败: {e}", exc_info=True)
            return {"success": False, "message": f"❌ 删除失败: {e}"}

    # ================================================================
    # 模板骨骼演示视频（问题4）
    #
    # 存储位置: data/templates/<action>/<template>.demo.mp4
    #   → 与模板 JSON 并列，跟随模板一起被 delete_template 清理
    # ================================================================

    def _template_demo_path(self, action: str, template_name: str) -> "Path":
        # 与模板 JSON 放在同一目录下，文件名固定 "<template>.demo.mp4"
        return self._library.root / action / f"{template_name}.demo.mp4"

    def _build_template_demo(
        self,
        action: str,
        template_name: str,
        sequence=None,
    ) -> Optional[str]:
        """内部方法：若不存在则真正渲染并保存演示视频"""
        from src.correction.template_video import render_template_demo_video

        out_path = self._template_demo_path(action, template_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if sequence is None:
            try:
                sequence = self._library.load_template(action, template_name)
            except Exception as e:
                logger.warning(f"加载模板失败: {e}")
                return None

        if sequence is None or sequence.num_frames == 0:
            return None

        return render_template_demo_video(
            sequence=sequence,
            output_path=str(out_path),
            title=f"{action} / {template_name}",
        )

    def get_template_demo_path(
        self,
        action: str,
        template_name: str,
        build_if_missing: bool = True,
    ) -> Optional[str]:
        """
        获取指定模板的演示视频路径。

        - 若视频已存在 → 直接返回路径（**点击即播放，无生成延时**）
        - 若不存在且 build_if_missing=True → 按需渲染一次后返回
        - 若不存在且 build_if_missing=False → 返回 None

        Args:
            action: 动作类别
            template_name: 模板名
            build_if_missing: 文件不存在时是否按需生成

        Returns:
            视频路径字符串；失败或不存在时 None
        """
        out_path = self._template_demo_path(action, template_name)
        if out_path.exists():
            return str(out_path)
        if not build_if_missing:
            return None
        return self._build_template_demo(action, template_name)

    # ================================================================
    # 帧对比查看器（问题5）
    #
    # 设计原则：
    # - 采用与 comparison_video 完全一致的 DTW 子序列匹配帧映射，
    #   保证滑条拖到任一用户帧时，右侧模板帧就是它"同步演示"的那一帧。
    # - 渲染复用 _draw_skeleton_on_region，与模板对比视频的视觉风格一致。
    # - 不做归一化 / 旋转，保证骨骼形态正常（不会出现扭曲）。
    # ================================================================

    def prepare_frame_viewer(self, report) -> Optional[dict]:
        """
        基于 CorrectionReport 构建帧对比所需的"索引表"。

        一次性地：
          1. 在原始用户序列 + 原始模板序列上跑子序列 DTW，得到 {user_frame_idx: tpl_frame_idx}
          2. 记录用户 PoseSequence 帧的 seq_index→frame_index 映射
          3. 返回一个 dict，供后续多次"按用户帧渲染"复用

        Args:
            report: CorrectionReport（需带 _comparison_result / _best_template）

        Returns:
            dict {
              "user_seq": PoseSequence,
              "template_seq": PoseSequence,
              "dtw_map": Dict[int_user_seq_idx, int_tpl_seq_idx],
              "match_start": int,   # user_seq 本地索引
              "match_end": int,
              "joint_deviations": Dict[str, float],  # 中/英文关节 → 偏差
              "total_user_frames": int,
            } 或 None
        """
        comp = getattr(report, "_comparison_result", None)
        tpl_seq = getattr(report, "_best_template", None)
        if comp is None or tpl_seq is None:
            return None

        # 我们需要"原始"用户序列用于渲染（未预处理，骨骼形态正常）
        user_seq = getattr(report, "_user_sequence", None)
        if user_seq is None:
            user_seq = getattr(comp, "user_sequence", None)
        if user_seq is None:
            # 最后兜底：用预处理后的
            user_seq = getattr(comp, "processed_query", None)
        if user_seq is None or user_seq.num_frames == 0:
            return None
        if tpl_seq.num_frames == 0:
            return None

        from src.action_comparison.comparison_video import _build_direct_dtw_map
        from src.correction.template_video import compute_sequence_fit_box

        dtw_map, match_start, match_end = _build_direct_dtw_map(
            user_seq, tpl_seq
        )

        # 预计算居中缩放参数（整段序列共用，渲染时保持稳定不抖）
        user_fit_box = compute_sequence_fit_box(user_seq)
        template_fit_box = compute_sequence_fit_box(tpl_seq)

        # 让两侧骨骼显示大小一致：统一用较大 scale_ref（更小的放大系数）
        # 这样动作较小的一方不会被放太大、动作较大的一方也能完整装进画布。
        # fit_box = (cx, cy, half_w, half_h, scale_ref)
        unified_scale = max(user_fit_box[4], template_fit_box[4])
        user_fit_box = (*user_fit_box[:4], unified_scale)
        template_fit_box = (*template_fit_box[:4], unified_scale)

        return {
            "user_seq": user_seq,
            "template_seq": tpl_seq,
            "dtw_map": dtw_map,
            "match_start": match_start,
            "match_end": match_end,
            "user_fit_box": user_fit_box,
            "template_fit_box": template_fit_box,
            "joint_deviations": dict(report.joint_deviations or {}),
            "total_user_frames": user_seq.num_frames,
        }

    def render_viewer_frame(
        self,
        viewer_state: dict,
        user_frame_idx: int,
        width: int = 480,
        height: int = 640,
    ) -> Optional[dict]:
        """
        按"用户帧序号"渲染一帧并排骨骼对比图。

        Args:
            viewer_state: prepare_frame_viewer 返回的字典
            user_frame_idx: 用户序列中的帧号（0-based；这是 PoseSequence 内的
                           本地序号，因为播放时我们是逐帧扫 PoseSequence 而
                           非原始视频文件）
            width/height: 单侧画幅

        Returns:
            dict {
              "image": RGB ndarray,
              "step_info": str,
              "joint_table": [(中文名, 英文名, 偏差), ...]
              "advice_lines": [str, ...]
            }
        """
        if not viewer_state:
            return None

        from src.correction.template_video import render_pair_frame
        from src.action_comparison.distance_metrics import (
            CORE_JOINT_NAMES, CORE_JOINT_INDICES,
        )
        from src.pose_estimation.data_types import LANDMARK_NAMES

        user_seq = viewer_state["user_seq"]
        tpl_seq = viewer_state["template_seq"]
        dtw_map = viewer_state["dtw_map"]
        match_start = viewer_state["match_start"]
        match_end = viewer_state["match_end"]

        user_total = user_seq.num_frames
        tpl_total = tpl_seq.num_frames
        # 夹紧到合法范围
        uidx = max(0, min(int(user_frame_idx), user_total - 1))

        # 取对应模板帧（三阶段：准备 / 匹配 / 收尾）
        if uidx < match_start:
            tidx = 0
            phase = "准备"
        elif uidx > match_end:
            tidx = tpl_total - 1
            phase = "收尾"
        else:
            if uidx in dtw_map:
                tidx = dtw_map[uidx]
            else:
                keys = sorted(dtw_map.keys()) if dtw_map else []
                if keys:
                    nearest = min(keys, key=lambda k: abs(k - uidx))
                    tidx = dtw_map[nearest]
                else:
                    # 无 DTW → 线性映射
                    if match_end > match_start:
                        ratio = (uidx - match_start) / (match_end - match_start)
                        tidx = int(ratio * (tpl_total - 1))
                    else:
                        tidx = 0
            phase = "同步"
        tidx = max(0, min(tidx, tpl_total - 1))

        user_frame = user_seq.frames[uidx]
        tpl_frame = tpl_seq.frames[tidx]

        # 关节偏差排序（用已有 joint_deviations，同步标注红圈）
        deviations = viewer_state.get("joint_deviations", {})
        # 英文名 → 索引
        name_to_idx = {n: i for i, n in enumerate(LANDMARK_NAMES)}
        joint_rows = []
        for en_name, dev in deviations.items():
            idx = name_to_idx.get(en_name)
            if idx is None:
                continue
            cn = CORE_JOINT_NAMES.get(idx, en_name)
            joint_rows.append((cn, en_name, float(dev), idx))
        joint_rows.sort(key=lambda x: x[2], reverse=True)

        # 高亮前 3 个偏差较大的关节
        highlight = [r[3] for r in joint_rows[:3] if r[2] >= 0.08]

        img_bgr = render_pair_frame(
            user_frame=user_frame,
            template_frame=tpl_frame,
            user_fit_box=viewer_state.get("user_fit_box"),
            template_fit_box=viewer_state.get("template_fit_box"),
            width=width,
            height=height,
            title_left="Your Action",
            title_right="Template",
            highlight_joints=highlight,
        )
        import cv2 as _cv2
        img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)

        # 针对"本帧"的建议：挑出当前帧偏差大的关节名做提示
        advice = []
        for cn, _en, dev, _idx in joint_rows[:3]:
            if dev < 0.08:
                continue
            level = "严重" if dev >= 0.18 else "中等"
            advice.append(
                f"• **{cn}**（偏差 {dev:.3f}，{level}）：请将 {cn} 向模板位置靠近。"
            )
        if not advice:
            advice.append("✅ 本帧整体吻合，继续保持！")

        step_info = (
            f"用户帧 **{uidx + 1}/{user_total}**  ↔  "
            f"模板帧 **{tidx + 1}/{tpl_total}**  ·  阶段：{phase}"
        )

        return {
            "image": img_rgb,
            "step_info": step_info,
            "joint_table": [(cn, en, round(d, 4)) for cn, en, d, _ in joint_rows],
            "advice_lines": advice,
            "total_user_frames": user_total,
        }

    # ================================================================
    # 内部方法
    # ================================================================

    def _analyze_sequence_internal(
        self,
        sequence: PoseSequence,
        action_name: Optional[str],
        progress_callback: Optional[Callable],
        step_offset: int = 0,
        video_path: Optional[str] = None,
    ) -> AnalysisResult:
        """
        内部分析方法（从 PoseSequence 开始）

        Args:
            sequence: 骨骼序列
            action_name: 动作类别
            progress_callback: 进度回调
            step_offset: 步骤偏移量（视频模式偏移 1）
            video_path: 原始视频路径（用于生成对比视频，None 则跳过）
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

            # 确定使用的算法名称
            algo_used = self._resolve_offline_algorithm()
            algo_display = ALGORITHM_DISPLAY_NAMES.get(algo_used, algo_used)

            # 生成可视化文件
            deviation_plot_path = None
            try:
                plot_path = self._output_dir / f"deviation_{int(time.time())}.png"
                self._visualizer.plot_deviation_bar(report, save_path=str(plot_path))
                deviation_plot_path = str(plot_path)
            except Exception as e:
                logger.warning(f"生成偏差图失败: {e}")

            # 生成骨骼对比视频
            comparison_video_path = None
            if video_path and report._best_template and report._comparison_result:
                try:
                    video_out = self._output_dir / f"comparison_{int(time.time())}.mp4"
                    if progress_callback:
                        progress_callback(
                            step_offset + 4, 5,
                            "正在生成骨骼对比视频…"
                        )
                    comparison_video_path = generate_comparison_video(
                        video_path=video_path,
                        user_sequence=sequence,
                        template_sequence=report._best_template,
                        comparison_result=report._comparison_result,
                        joint_deviations=report.joint_deviations,
                        output_path=str(video_out),
                        quality_score=report.quality_score,
                        corrections=list(report.corrections),
                        progress_callback=None,
                    )
                except Exception as e:
                    logger.warning(f"生成对比视频失败: {e}")

            # 构建结果
            display_name = ACTION_DISPLAY_NAMES.get(
                report.action_name, report.action_display_name or report.action_name
            )

            result = AnalysisResult(
                action_name=report.action_name,
                action_display_name=display_name,
                quality_score=report.quality_score,
                similarity=report.similarity,
                report_text=report.to_text() + f"\n\n【使用算法】{algo_display}",
                deviation_plot_path=deviation_plot_path,
                skeleton_video_path=None,
                comparison_video_path=comparison_video_path,
                corrections=list(report.corrections),
                report=report,
                confidence=report.confidence,
                algorithm_used=algo_used,
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

    # ================================================================
    # 算法选择
    # ================================================================

    @property
    def algorithm(self) -> str:
        """当前 DTW 算法设置"""
        return self._algorithm

    def set_algorithm(self, algorithm: str):
        """
        设置 DTW 算法

        Args:
            algorithm: "dtw" / "fastdtw" / "ddtw" / "auto"
        """
        valid = {"dtw", "fastdtw", "ddtw", "auto"}
        if algorithm not in valid:
            raise ValueError(f"不支持的算法: {algorithm}，可选: {valid}")

        self._algorithm = algorithm

        # 更新 CorrectionPipeline 的对比器
        offline_algo = self._resolve_offline_algorithm()
        offline_metric = self._resolve_offline_metric()
        self._correction_pipeline._comparator._algorithm = offline_algo
        self._correction_pipeline._comparator._metric = offline_metric

        # 更新实时引擎（如果已创建）
        if self._realtime_engine is not None:
            self._realtime_engine.algorithm = self._resolve_realtime_algorithm()

        logger.info(
            f"DTW 算法切换: {algorithm} "
            f"(离线={offline_algo}+{offline_metric}, "
            f"实时={self._resolve_realtime_algorithm()})"
        )

    def _resolve_offline_algorithm(self) -> str:
        """解析离线分析使用的实际算法"""
        if self._algorithm == "auto":
            return "dtw"  # 角度特征 + 经典DTW 区分度最好
        return self._algorithm

    def _resolve_offline_metric(self) -> str:
        """解析离线分析使用的距离度量"""
        if self._algorithm == "auto":
            return "euclidean"  # 角度特征下欧氏距离最稳定
        return "euclidean"

    def _resolve_realtime_algorithm(self) -> str:
        """解析实时分析使用的实际算法"""
        if self._algorithm == "auto":
            return "fastdtw"
        return self._algorithm

    # ================================================================
    # 实时反馈
    # ================================================================

    def init_realtime_session(self, action_name: str) -> bool:
        """
        初始化实时反馈会话

        根据动作名称加载模板序列，创建 RealtimeFeedbackEngine。

        Args:
            action_name: 动作类别名称

        Returns:
            是否成功（模板库中有该动作的模板时返回 True）
        """
        templates = self._library.load_all_templates(action_name)
        if not templates:
            logger.warning(f"动作 '{action_name}' 无可用模板，无法启动实时反馈")
            return False

        # 使用第一个模板作为参考
        template_name, template_seq = next(iter(templates.items()))

        realtime_algo = self._resolve_realtime_algorithm()
        self._realtime_engine = RealtimeFeedbackEngine(
            template_sequence=template_seq,
            algorithm=realtime_algo,
            window_size=10,
        )

        logger.info(
            f"实时反馈会话初始化: action={action_name}, "
            f"template={template_name}, algorithm={realtime_algo}"
        )
        return True

    def process_realtime_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        expected_total_frames: int,
    ) -> tuple:
        """
        处理实时模式的单帧

        Args:
            frame: BGR 图像 (H, W, 3)
            frame_index: 当前帧索引
            expected_total_frames: 预期总帧数

        Returns:
            (ProcessedFrame, FeedbackSnapshot or None)
        """
        if frame is None or frame.size == 0:
            return ProcessedFrame(annotated_image=frame), None

        streamer = self._ensure_rt_streamer()
        raw = streamer.process(frame)
        landmarks = self._rt_smoother.update(raw)

        if landmarks is None:
            empty_snap = FeedbackSnapshot(has_pose=False)
            return ProcessedFrame(annotated_image=frame.copy()), empty_snap

        # 用平滑后的 landmarks 构造 PoseFrame 绘制
        draw_frame = self._landmarks_to_poseframe(landmarks)
        annotated = frame.copy()
        draw_skeleton(annotated, draw_frame)

        # 实时反馈分析
        snapshot = None
        if self._realtime_engine is not None:
            snapshot = self._realtime_engine.analyze_frame(
                landmarks=landmarks,
                frame_index=frame_index,
                expected_total_frames=expected_total_frames,
            )

        return ProcessedFrame(
            annotated_image=annotated,
            landmarks=landmarks,
        ), snapshot

    def end_realtime_session(self):
        """结束实时反馈会话"""
        if self._realtime_engine is not None:
            self._realtime_engine.reset()
            self._realtime_engine = None

    def close(self):
        """释放资源"""
        if hasattr(self, '_estimator') and self._estimator:
            self._estimator.close()
        if hasattr(self, '_cam_streamer') and self._cam_streamer:
            self._cam_streamer.close()
            self._cam_streamer = None
        if hasattr(self, '_rt_streamer') and self._rt_streamer:
            self._rt_streamer.close()
            self._rt_streamer = None
