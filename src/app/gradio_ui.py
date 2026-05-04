"""
Gradio Web 交互界面

基于 Gradio Blocks 的多 Tab 布局：
- Tab 1: 📹 视频分析 — 上传视频或摄像头录制 → 矫正报告
- Tab 2: 📷 实时模式 — 倒计时 → 实时跟做 → 即时反馈 → 总体报告
- Tab 3: 📋 模板管理 — 查看/录入标准动作模板
- Tab 4: ℹ️ 系统说明 — 使用指南和技术架构
"""

import logging
import math
import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from .pipeline import (
    AppPipeline,
    ACTION_DISPLAY_NAMES,
    ALGORITHM_CHOICES,
    ALGORITHM_DISPLAY_NAMES,
)
from .session import SessionManager, RecordingState

logger = logging.getLogger(__name__)

# 实时模式性能参数
_PROCESS_WIDTH = 320   # 推理用缩放宽度（越小越快）
_DISPLAY_WIDTH = 480   # 返回给前端的显示宽度（降低传输量）

# 算法下拉框选项标签列表
_ALGO_LABELS = [label for label, _ in ALGORITHM_CHOICES]
# 标签 → 算法 ID 映射
_ALGO_LABEL_TO_ID = {label: algo_id for label, algo_id in ALGORITHM_CHOICES}


def create_gradio_app(pipeline: AppPipeline) -> gr.Blocks:
    """
    创建 Gradio 应用

    Args:
        pipeline: AppPipeline 实例

    Returns:
        gr.Blocks 应用实例
    """
    session = SessionManager(template_library=pipeline.template_library)

    # ================================================================
    # 辅助函数
    # ================================================================

    def _resolve_algorithm(algo_label: str) -> str:
        """从下拉框标签解析算法 ID"""
        return _ALGO_LABEL_TO_ID.get(algo_label, "auto")

    def _resolve_action(action_choice: str) -> Optional[str]:
        """从下拉框解析动作名称，"自动识别"返回 None"""
        return None if action_choice == "自动识别" else action_choice

    def _scale_frame(frame: np.ndarray):
        """缩放帧用于推理，返回 (缩小帧, 缩放比例)"""
        if frame is None:
            return frame, 1.0
        h, w = frame.shape[:2]
        if w > _PROCESS_WIDTH:
            scale = _PROCESS_WIDTH / w
            small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            return small, scale
        return frame, 1.0

    def _shrink_for_display(image: np.ndarray) -> np.ndarray:
        """
        缩小图像用于前端显示，降低 Gradio 传输数据量。
        这是提升帧率的关键优化。
        """
        if image is None:
            return image
        h, w = image.shape[:2]
        if w > _DISPLAY_WIDTH:
            scale = _DISPLAY_WIDTH / w
            return cv2.resize(image, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        return image

    # ================================================================
    # Tab 1 回调：视频分析
    # ================================================================

    def on_analyze_video(video_file, action_choice, algo_label, progress=gr.Progress()):
        """上传视频分析"""
        # gr.File 可能返回 str 路径，也可能返回 dict（含 name 键）
        if video_file is None:
            return "⚠️ 请先上传视频文件", None, None, None
        if isinstance(video_file, dict):
            video_path = video_file.get("name", "")
        else:
            video_path = str(video_file)
        if not video_path:
            return "⚠️ 请先上传视频文件", None, None, None

        action_name = _resolve_action(action_choice)
        algo_id = _resolve_algorithm(algo_label)
        pipeline.set_algorithm(algo_id)

        def progress_cb(step, total, msg):
            progress(step / total, desc=msg)

        progress(0, desc="开始分析…")
        result = pipeline.analyze_video(
            video_path=video_path,
            action_name=action_name,
            progress_callback=progress_cb,
        )
        session.last_result = result
        return result.report_text, result.deviation_plot_path, result.summary(), result.comparison_video_path

    def on_start_cam_recording():
        """视频分析 Tab — 摄像头录制：开始"""
        pipeline.reset_camera_smoothers()
        session.start_recording()
        return (
            "🔴 录制中… 执行动作后点击「停止录制」",
            gr.update(interactive=False),
            gr.update(interactive=True),
        )

    def on_cam_record_frame(frame):
        """视频分析 Tab — 摄像头录制：逐帧处理"""
        if frame is None:
            return None
        small, _ = _scale_frame(frame)
        processed = pipeline.process_camera_frame(small)
        if session.is_recording and processed.has_pose:
            session.add_frame(processed.landmarks)
        return _shrink_for_display(processed.annotated_image)

    def on_stop_cam_recording(action_choice, algo_label):
        """视频分析 Tab — 摄像头录制：停止并分析"""
        sequence = session.stop_recording()
        if sequence is None:
            session.finish_analysis()
            return (
                "⚠️ 录制帧数不足（至少 5 帧），请确保全身在画面中",
                "", None, None,
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        action_name = _resolve_action(action_choice)
        algo_id = _resolve_algorithm(algo_label)
        pipeline.set_algorithm(algo_id)

        # fallback
        if action_name is None:
            actions = session.get_action_list()
            if actions:
                action_name = actions[0]

        result = pipeline.analyze_sequence(
            sequence_data=sequence, action_name=action_name
        )
        session.finish_analysis(result)
        return (
            "✅ 分析完成",
            result.report_text,
            result.deviation_plot_path,
            result.comparison_video_path,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    # ================================================================
    # Tab 2 回调：实时模式
    # ================================================================

    def on_start_realtime(action_choice, algo_label):
        """开始实时模式：倒计时 → 实时分析"""
        action_name = _resolve_action(action_choice)
        algo_id = _resolve_algorithm(algo_label)
        pipeline.set_algorithm(algo_id)

        # fallback
        if action_name is None:
            actions = session.get_action_list()
            if actions:
                action_name = actions[0]
            else:
                return (
                    "⚠️ 请先选择动作类型，或在模板管理中录入模板",
                    "",
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                )

        # 初始化实时引擎
        success = pipeline.init_realtime_session(action_name)
        if not success:
            return (
                f"⚠️ 动作 '{action_name}' 无可用模板，请先录入模板",
                "",
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        # 重置平滑器，开始倒计时
        pipeline.reset_camera_smoothers()
        session.start_countdown()
        return (
            "⏱️ 3... 准备好！",
            "等待倒计时结束…",
            gr.update(interactive=False),
            gr.update(interactive=True),
        )

    def on_realtime_frame(frame):
        """
        实时模式帧处理：
        - COUNTDOWN 状态：显示倒计时，不分析
        - REALTIME 状态：骨骼叠加 + 固定窗口 DTW + 实时建议
        - 其他状态：仅骨骼叠加

        性能优化：直接用缩小后的图像返回前端，不放大回原尺寸，
        减少 Gradio 传输数据量是提升帧率的关键。
        """
        if frame is None:
            return None, gr.update(), gr.update()

        # 缩放用于推理（关键：不再放大回原尺寸）
        small, _ = _scale_frame(frame)

        # ---- 倒计时状态（帧驱动）----
        if session.is_countdown:
            remaining_frames = session.advance_countdown()
            if session.countdown_finished:
                session.start_realtime()
                status = "🟢 开始跟做！"
                feedback_md = "**实时分析已启动，请开始做动作…**"
            else:
                # 每10帧约1秒，显示秒数
                remaining_sec = math.ceil(remaining_frames * 0.1)
                status = f"⏱️ {remaining_sec}..."
                feedback_md = f"## ⏱️ 倒计时: {remaining_sec}\n\n请准备好姿势"

            processed = pipeline.process_camera_frame(small)
            display = _shrink_for_display(
                processed.annotated_image if processed.annotated_image is not None else small
            )
            return display, status, feedback_md

        # ---- 实时分析状态 ----
        if session.is_realtime:
            frame_idx = session.advance_realtime_frame()
            expected_total = 90  # 约 3 秒 @30fps

            processed, snapshot = pipeline.process_realtime_frame(
                small, frame_idx, expected_total
            )

            display = _shrink_for_display(
                processed.annotated_image if processed.annotated_image is not None else small
            )

            # 缓存 landmarks
            if processed.has_pose:
                session.add_frame(processed.landmarks)

            # 更新反馈
            if snapshot is not None:
                session.last_feedback = snapshot
                feedback_md = snapshot.to_markdown()
            else:
                feedback_md = "⚠️ 未检测到姿态"

            frame_count = session.frame_count
            status = f"🟢 实时分析中 — 已采集 {frame_count} 帧"

            return display, status, feedback_md

        # ---- 空闲/其他状态 ----
        processed = pipeline.process_camera_frame(small)
        display = _shrink_for_display(
            processed.annotated_image if processed.annotated_image is not None else small
        )
        return display, gr.update(), gr.update()

    def on_stop_realtime(action_choice, algo_label):
        """停止实时分析，生成总体报告"""
        sequence = session.stop_realtime()
        pipeline.end_realtime_session()

        if sequence is None:
            session.finish_analysis()
            return (
                "⚠️ 帧数不足（至少 5 帧），请重试",
                "", None,
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        action_name = _resolve_action(action_choice)
        algo_id = _resolve_algorithm(algo_label)
        pipeline.set_algorithm(algo_id)

        if action_name is None:
            actions = session.get_action_list()
            if actions:
                action_name = actions[0]

        result = pipeline.analyze_sequence(
            sequence_data=sequence, action_name=action_name
        )
        session.finish_analysis(result)

        return (
            "✅ 分析完成 — 总体报告已生成",
            result.report_text,
            result.deviation_plot_path,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    # ================================================================
    # Tab 3 回调：模板管理
    # ================================================================

    def on_refresh_templates():
        session.refresh_action_list()
        info = pipeline.get_template_info()
        if not info:
            return "📭 模板库为空，请录入标准动作模板。"
        lines = ["## 📋 模板库概览\n"]
        for item in info:
            lines.append(
                f"- **{item['display_name']}** (`{item['action']}`): "
                f"{item['count']} 个模板"
            )
            for t in item['templates']:
                lines.append(f"  - {t}")
        return "\n".join(lines)

    def on_record_template(video_file, action_input):
        # gr.File 可能返回 str 路径，也可能返回 dict（含 name 键）
        if video_file is None:
            return "⚠️ 请上传视频文件", gr.update(), gr.update()
        if isinstance(video_file, dict):
            video_path = video_file.get("name", "")
        else:
            video_path = str(video_file)
        if not video_path:
            return "⚠️ 请上传视频文件", gr.update(), gr.update()
        if not action_input or not action_input.strip():
            return "⚠️ 请输入动作名称", gr.update(), gr.update()
        action_name = action_input.strip()
        result = pipeline.record_template_with_error(video_path, action_name)
        if result.success:
            session.refresh_action_list()
            new_choices = session.get_action_choices()
            return (
                f"✅ 模板录入成功: {action_name}",
                gr.update(choices=new_choices),
                gr.update(choices=new_choices),
            )
        return f"❌ 模板录入失败: {result.error}", gr.update(), gr.update()

    # -- 模板管理：删除模板 --

    def _list_actions_with_templates():
        """返回包含模板的动作列表（用于删除下拉框）"""
        return [it["action"] for it in pipeline.get_template_info() if it["count"] > 0]

    def on_del_refresh_actions():
        """刷新删除面板的动作下拉框"""
        actions = _list_actions_with_templates()
        default = actions[0] if actions else None
        templates = pipeline.template_library.list_templates(default) if default else []
        return (
            gr.update(choices=actions, value=default),
            gr.update(choices=templates,
                     value=(templates[0] if templates else None)),
        )

    def on_del_action_change(action_name):
        """动作变化时刷新模板下拉框"""
        if not action_name:
            return gr.update(choices=[], value=None)
        templates = pipeline.template_library.list_templates(action_name)
        return gr.update(choices=templates,
                         value=(templates[0] if templates else None))

    # -- 模板管理：骨骼演示视频（问题4） --

    def on_tpl_preview_refresh():
        """刷新预览面板的动作/模板下拉"""
        actions = _list_actions_with_templates()
        default = actions[0] if actions else None
        templates = (pipeline.template_library.list_templates(default)
                     if default else [])
        first_tpl = templates[0] if templates else None
        # 直接显示第一个模板的演示视频（若已缓存/可生成）
        video = (pipeline.get_template_demo_path(default, first_tpl)
                 if (default and first_tpl) else None)
        return (
            gr.update(choices=actions, value=default),
            gr.update(choices=templates, value=first_tpl),
            video,
        )

    def on_tpl_preview_action_change(action_name):
        """预览面板：动作切换时更新模板下拉，并自动加载第一个模板的视频"""
        if not action_name:
            return gr.update(choices=[], value=None), None
        templates = pipeline.template_library.list_templates(action_name)
        first_tpl = templates[0] if templates else None
        video = (pipeline.get_template_demo_path(action_name, first_tpl)
                 if first_tpl else None)
        return (
            gr.update(choices=templates, value=first_tpl),
            video,
        )

    def on_tpl_preview_template_change(action_name, template_name):
        """预览面板：模板切换时自动加载对应演示视频（点击即播放）"""
        if not action_name or not template_name:
            return None
        return pipeline.get_template_demo_path(action_name, template_name)

    # -- 视频分析：帧对比查看器（问题5） --

    def on_frame_viewer_init():
        """
        分析完成后：
          1. 构建 DTW 帧映射索引（保存到 session）
          2. 配置滑条范围 = 用户原始帧数
          3. 立即渲染第一帧
        """
        res = getattr(session, "last_result", None)
        if res is None or res.report is None:
            return (
                gr.update(minimum=0, maximum=0, value=0,
                          label="对齐步骤（无可用数据）", visible=False),
                None, "", "", "",
                gr.update(visible=False),
            )
        viewer_state = pipeline.prepare_frame_viewer(res.report)
        if viewer_state is None:
            session.frame_viewer_state = None
            return (
                gr.update(minimum=0, maximum=0, value=0,
                          label="对齐步骤（该分析不支持帧对比）", visible=False),
                None, "", "", "",
                gr.update(visible=False),
            )
        session.frame_viewer_state = viewer_state
        total = viewer_state["total_user_frames"]
        # 渲染第一帧
        data = pipeline.render_viewer_frame(viewer_state, 0)
        if data is None:
            return (
                gr.update(minimum=1, maximum=total, value=1, step=1,
                          label=f"用户帧（共 {total} 帧）", visible=True),
                None, "", "", "",
                gr.update(visible=True),
            )
        tbl_md = _joint_table_md(data["joint_table"])
        advice_md = "\n\n".join(data["advice_lines"])
        return (
            gr.update(minimum=1, maximum=total, value=1, step=1,
                      label=f"用户帧（共 {total} 帧） — 拖动查看动作对比",
                      visible=True),
            data["image"], data["step_info"], tbl_md, advice_md,
            gr.update(visible=True),
        )

    def on_frame_viewer_render(user_frame_1based):
        """滑条变化：渲染指定用户帧的并排骨骼对比"""
        viewer_state = getattr(session, "frame_viewer_state", None)
        if viewer_state is None:
            return None, "❌ 尚无分析结果", "", ""
        try:
            uidx = int(user_frame_1based) - 1
        except (TypeError, ValueError):
            uidx = 0
        data = pipeline.render_viewer_frame(viewer_state, uidx)
        if data is None:
            return None, "❌ 渲染失败", "", ""
        tbl_md = _joint_table_md(data["joint_table"])
        advice_md = "\n\n".join(data["advice_lines"])
        return data["image"], data["step_info"], tbl_md, advice_md

    def _joint_table_md(rows):
        """将 [(中文, 英文, 偏差), ...] 渲染为 markdown 表格"""
        lines = ["| 关节 | 英文名 | 偏差 |", "|---|---|---|"]
        for cn, en, dev in rows[:12]:
            mark = "🔴" if dev >= 0.18 else ("🟡" if dev >= 0.08 else "🟢")
            lines.append(f"| {mark} {cn} | `{en}` | {dev:.4f} |")
        return "\n".join(lines)

    def on_delete_template(action_name, template_name):
        """执行删除"""
        if not action_name:
            return (
                "⚠️ 请选择动作类别",
                gr.update(), gr.update(), gr.update(), gr.update(),
            )
        if not template_name:
            return (
                "⚠️ 请选择要删除的模板",
                gr.update(), gr.update(), gr.update(), gr.update(),
            )
        result = pipeline.delete_template(action_name, template_name,
                                          auto_remove_empty_action=True)

        # 刷新会话动作列表与所有下拉框
        session.refresh_action_list()
        action_choices = session.get_action_choices()

        # 删除面板的下拉框
        remain_actions = _list_actions_with_templates()
        new_action = (action_name
                      if action_name in remain_actions
                      else (remain_actions[0] if remain_actions else None))
        remain_templates = (pipeline.template_library.list_templates(new_action)
                            if new_action else [])

        return (
            result["message"],
            gr.update(choices=remain_actions, value=new_action),
            gr.update(choices=remain_templates,
                      value=(remain_templates[0] if remain_templates else None)),
            # 同步视频分析 & 实时模式的动作下拉
            gr.update(choices=action_choices),
            gr.update(choices=action_choices),
        )

    def on_delete_action(action_name):
        """删除整个动作类别（及所有模板）"""
        if not action_name:
            return (
                "⚠️ 请选择动作类别",
                gr.update(), gr.update(), gr.update(), gr.update(),
            )
        result = pipeline.delete_action(action_name)
        session.refresh_action_list()
        action_choices = session.get_action_choices()
        remain_actions = _list_actions_with_templates()
        new_action = remain_actions[0] if remain_actions else None
        remain_templates = (pipeline.template_library.list_templates(new_action)
                            if new_action else [])
        return (
            result["message"],
            gr.update(choices=remain_actions, value=new_action),
            gr.update(choices=remain_templates,
                      value=(remain_templates[0] if remain_templates else None)),
            gr.update(choices=action_choices),
            gr.update(choices=action_choices),
        )

    # -- 模板管理：摄像头录制模板 --

    def on_tpl_cam_frame(frame):
        """模板管理 — 摄像头录制帧处理"""
        if frame is None:
            return None
        small, _ = _scale_frame(frame)
        processed = pipeline.process_camera_frame(small)
        if session.is_recording and processed.has_pose:
            session.add_frame(processed.landmarks)
        return _shrink_for_display(processed.annotated_image)

    def on_tpl_start_recording():
        """模板管理 — 开始录制"""
        pipeline.reset_camera_smoothers()
        session.start_recording()
        return (
            "🔴 录制中… 请执行标准动作，完成后点击「停止录制」",
            gr.update(interactive=False),
            gr.update(interactive=True),
        )

    def on_tpl_stop_recording(action_input):
        """模板管理 — 停止录制并保存为模板"""
        if not action_input or not action_input.strip():
            session.finish_analysis()
            return (
                "⚠️ 请先输入动作名称",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(), gr.update(),
            )

        sequence = session.stop_recording()
        if sequence is None:
            session.finish_analysis()
            return (
                "⚠️ 录制帧数不足（至少 5 帧），请确保全身在画面中",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(), gr.update(),
            )

        action_name = action_input.strip()

        # 将 numpy 数组转为 PoseSequence 然后保存
        from src.pose_estimation.data_types import PoseLandmark, PoseFrame, PoseSequence
        seq = PoseSequence(fps=10.0)
        for i in range(sequence.shape[0]):
            landmarks = []
            for j in range(sequence.shape[1]):
                landmarks.append(PoseLandmark(
                    x=float(sequence[i, j, 0]),
                    y=float(sequence[i, j, 1]),
                    z=float(sequence[i, j, 2]),
                    visibility=float(sequence[i, j, 3]) if sequence.shape[2] > 3 else 1.0,
                ))
            frame = PoseFrame(
                timestamp=i / 10.0,
                frame_index=i,
                landmarks=landmarks,
            )
            seq.add_frame(frame)

        import time as _time
        template_name = f"cam_template_{int(_time.time())}"

        display_name = ACTION_DISPLAY_NAMES.get(action_name, action_name)
        if action_name not in pipeline.template_library.list_actions():
            pipeline.template_library.add_action(action_name, display_name)

        pipeline.template_library.add_template(action_name, seq, template_name)
        session.finish_analysis()
        session.refresh_action_list()
        new_choices = session.get_action_choices()

        return (
            f"✅ 模板录入成功: {action_name}/{template_name} ({sequence.shape[0]} 帧)",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(choices=new_choices),
            gr.update(choices=new_choices),
        )

    # ================================================================
    # 界面构建
    # ================================================================

    with gr.Blocks(title="人体动作矫正系统") as app:

        gr.Markdown("# 🏋️ 基于深度学习的人体动作矫正系统")
        gr.Markdown("*CS2201 张玉倍 — 本科毕业设计*")

        # ============================================================
        # Tab 1: 视频分析
        # ============================================================
        with gr.Tab("📹 视频分析"):
            gr.Markdown("### 上传视频或摄像头录制，获取动作矫正报告")
            gr.Markdown(
                "> 💡 支持格式: MP4、AVI、MOV、WebM、MKV 等。"
                "后端使用 OpenCV 处理，浏览器预览不影响分析。"
                "如遇编码问题，建议转为 MP4 (H.264)。"
            )

            with gr.Row():
                vid_input_mode = gr.Radio(
                    choices=["上传视频", "摄像头录制"],
                    value="上传视频",
                    label="输入方式",
                )
                vid_action_dropdown = gr.Dropdown(
                    choices=session.get_action_choices(),
                    value=(
                        session.get_action_choices()[1]
                        if len(session.get_action_choices()) > 1
                        else "自动识别"
                    ),
                    label="动作类型",
                )
                vid_algo_dropdown = gr.Dropdown(
                    choices=_ALGO_LABELS,
                    value=_ALGO_LABELS[-1],  # "自动选择（推荐）"
                    label="DTW 算法",
                )

            # -- 上传视频子模式 --
            with gr.Column(visible=True) as upload_col:
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.File(
                            label="上传视频", file_types=["video"]
                        )
                        analyze_btn = gr.Button(
                            "🔍 开始分析", variant="primary", size="lg"
                        )
                    with gr.Column(scale=2):
                        vid_summary = gr.Textbox(
                            label="📊 分析摘要", lines=5, interactive=False
                        )
                        vid_report = gr.Textbox(
                            label="📝 矫正报告", lines=12, interactive=False
                        )
                        vid_plot = gr.Image(
                            label="📈 偏差分析图", type="filepath"
                        )
                        vid_comparison = gr.Video(
                            label="🎬 骨骼对比视频（左：您的动作 🟢🟡🔴 | 右：标准模板）",
                            height=360,
                        )

                # ===== 帧对比查看器（问题5） =====
                with gr.Group(visible=False) as frame_viewer_group:
                    gr.Markdown("### 🔍 帧对比查看器 — 拖动滑条查看每一帧的差异")
                    gr.Markdown(
                        "> 左侧为您的动作骨骼，右侧为模板骨骼。"
                        "红圈表示该帧偏差较大的关节。"
                        "下方会列出该帧的逐关节偏差以及针对性建议。"
                    )
                    with gr.Row():
                        frame_viewer_slider = gr.Slider(
                            minimum=0, maximum=0, value=0, step=1,
                            label="对齐步骤",
                            interactive=True,
                            visible=False,
                        )
                    with gr.Row():
                        with gr.Column(scale=2):
                            frame_viewer_image = gr.Image(
                                label="🦴 并排骨骼对比",
                                type="numpy",
                                interactive=False,
                                height=400,
                            )
                            frame_viewer_info = gr.Markdown("")
                        with gr.Column(scale=1):
                            frame_viewer_table = gr.Markdown(
                                "当前帧的关节偏差表会显示在这里。"
                            )
                            frame_viewer_advice = gr.Markdown(
                                "", label="针对该帧的建议",
                            )

            # -- 摄像头录制子模式 --
            with gr.Column(visible=False) as cam_record_col:
                with gr.Row():
                    with gr.Column(scale=1):
                        cam_rec_input = gr.Image(
                            label="摄像头",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            height=300,
                        )
                        cam_rec_display = gr.Image(
                            label="骨骼叠加",
                            type="numpy",
                            interactive=False,
                            height=300,
                        )
                        cam_rec_status = gr.Textbox(
                            label="状态",
                            value="⏸️ 空闲",
                            interactive=False,
                        )
                        with gr.Row():
                            cam_rec_start = gr.Button(
                                "🔴 开始录制", variant="primary"
                            )
                            cam_rec_stop = gr.Button(
                                "⏹️ 停止并分析",
                                variant="secondary",
                                interactive=False,
                            )
                    with gr.Column(scale=1):
                        cam_rec_report = gr.Textbox(
                            label="📝 矫正报告",
                            lines=12,
                            interactive=False,
                        )
                        cam_rec_plot = gr.Image(
                            label="📈 偏差分析图", type="filepath"
                        )

            # 模式切换
            def on_switch_input_mode(mode):
                return (
                    gr.update(visible=(mode == "上传视频")),
                    gr.update(visible=(mode == "摄像头录制")),
                )

            vid_input_mode.change(
                fn=on_switch_input_mode,
                inputs=[vid_input_mode],
                outputs=[upload_col, cam_record_col],
            )

            # 上传视频分析 — 分析完成后自动初始化帧对比查看器并渲染首帧
            analyze_btn.click(
                fn=on_analyze_video,
                inputs=[video_input, vid_action_dropdown, vid_algo_dropdown],
                outputs=[vid_report, vid_plot, vid_summary, vid_comparison],
            ).then(
                fn=on_frame_viewer_init,
                outputs=[
                    frame_viewer_slider,
                    frame_viewer_image,
                    frame_viewer_info,
                    frame_viewer_table,
                    frame_viewer_advice,
                    frame_viewer_group,
                ],
            )

            # 滑条拖动时刷新帧对比图
            frame_viewer_slider.change(
                fn=on_frame_viewer_render,
                inputs=[frame_viewer_slider],
                outputs=[
                    frame_viewer_image, frame_viewer_info,
                    frame_viewer_table, frame_viewer_advice,
                ],
                show_progress="hidden",
            )

            # 摄像头录制
            cam_rec_input.stream(
                fn=on_cam_record_frame,
                inputs=[cam_rec_input],
                outputs=[cam_rec_display],
                stream_every=0.1,
                show_progress="hidden",  # 禁止加载遮罩
            )
            cam_rec_start.click(
                fn=on_start_cam_recording,
                outputs=[cam_rec_status, cam_rec_start, cam_rec_stop],
            )
            cam_rec_stop.click(
                fn=on_stop_cam_recording,
                inputs=[vid_action_dropdown, vid_algo_dropdown],
                outputs=[
                    cam_rec_status,
                    cam_rec_report, cam_rec_plot, vid_comparison,
                    cam_rec_start, cam_rec_stop,
                ],
            ).then(
                fn=on_frame_viewer_init,
                outputs=[
                    frame_viewer_slider,
                    frame_viewer_image,
                    frame_viewer_info,
                    frame_viewer_table,
                    frame_viewer_advice,
                    frame_viewer_group,
                ],
            )

        # ============================================================
        # Tab 2: 实时模式
        # ============================================================
        with gr.Tab("📷 实时模式"):
            gr.Markdown(
                "### 实时跟做训练，边做边获得矫正提示\n"
                "**流程**: ① 点击摄像头区域开启摄像头 → "
                "② 选择动作和算法 → ③ 点击「开始跟做」→ "
                "3 秒倒计时 → 跟做动作（右侧实时显示建议）→ "
                "④ 点击「结束」→ 查看总体报告\n\n"
                "> ⚠️ **请先点击摄像头区域的录制按钮，看到画面后再点「开始跟做」**"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # 摄像头
                    rt_cam_input = gr.Image(
                        label="摄像头",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=300,
                    )
                    rt_cam_display = gr.Image(
                        label="骨骼叠加画面",
                        type="numpy",
                        interactive=False,
                        height=360,
                    )

                with gr.Column(scale=1):
                    rt_feedback = gr.Markdown(
                        value="**等待开始…**\n\n选择动作类型后点击「开始跟做」",
                        label="📊 实时矫正建议",
                    )

            rt_status = gr.Textbox(
                label="状态",
                value="⏸️ 空闲 — 请先打开摄像头",
                interactive=False,
            )

            with gr.Row():
                rt_action_dropdown = gr.Dropdown(
                    choices=session.get_action_choices(),
                    value=(
                        session.get_action_choices()[1]
                        if len(session.get_action_choices()) > 1
                        else "自动识别"
                    ),
                    label="动作类型",
                )
                rt_algo_dropdown = gr.Dropdown(
                    choices=_ALGO_LABELS,
                    value=_ALGO_LABELS[-1],
                    label="DTW 算法",
                )
                rt_start_btn = gr.Button(
                    "🔴 开始跟做", variant="primary"
                )
                rt_stop_btn = gr.Button(
                    "⏹️ 结束",
                    variant="secondary",
                    interactive=False,
                )

            gr.Markdown("---")
            gr.Markdown("### 📝 总体报告（结束后显示）")
            with gr.Row():
                with gr.Column():
                    rt_report = gr.Textbox(
                        label="总体矫正报告",
                        lines=12,
                        interactive=False,
                    )
                with gr.Column():
                    rt_plot = gr.Image(
                        label="📈 偏差分析图",
                        type="filepath",
                    )

            # 实时流处理
            rt_cam_input.stream(
                fn=on_realtime_frame,
                inputs=[rt_cam_input],
                outputs=[rt_cam_display, rt_status, rt_feedback],
                stream_every=0.1,
                show_progress="hidden",  # 禁止加载遮罩
            )

            # 开始按钮
            rt_start_btn.click(
                fn=on_start_realtime,
                inputs=[rt_action_dropdown, rt_algo_dropdown],
                outputs=[rt_status, rt_feedback, rt_start_btn, rt_stop_btn],
            )

            # 结束按钮
            rt_stop_btn.click(
                fn=on_stop_realtime,
                inputs=[rt_action_dropdown, rt_algo_dropdown],
                outputs=[
                    rt_status,
                    rt_report, rt_plot,
                    rt_start_btn, rt_stop_btn,
                ],
            )

        # ============================================================
        # Tab 3: 模板管理
        # ============================================================
        with gr.Tab("📋 模板管理"):
            gr.Markdown("### 管理标准动作模板库")

            with gr.Row():
                # ---- 左列：模板列表 ----
                with gr.Column(scale=1):
                    gr.Markdown("#### 📂 模板列表")
                    template_info = gr.Markdown('点击"刷新"查看模板列表')
                    refresh_btn = gr.Button("🔄 刷新列表")

                # ---- 右列：录入新模板 ----
                with gr.Column(scale=2):
                    gr.Markdown("#### ➕ 录入新模板")

                    tpl_name_input = gr.Textbox(
                        label="动作名称（必填）",
                        placeholder="例如: squat, arm_raise, lunge",
                        info="英文标识名，上传和录制共用",
                    )

                    with gr.Tabs():
                        # -- 方式一：上传视频 --
                        with gr.Tab("📁 上传视频"):
                            gr.Markdown(
                                "> 支持 MP4、AVI、MOV、WebM 等常见格式。"
                                "如遇格式问题，可用格式工厂等工具转为 MP4。"
                            )
                            tpl_video = gr.File(
                                label="上传标准动作视频",
                                file_types=["video"],
                            )
                            upload_btn = gr.Button(
                                "📥 上传并录入", variant="primary"
                            )
                            upload_status = gr.Textbox(
                                label="录入状态", interactive=False
                            )

                        # -- 方式二：摄像头录制 --
                        with gr.Tab("📷 摄像头录制"):
                            gr.Markdown(
                                "> 先点击摄像头开启画面，"
                                "然后点「开始录制」执行标准动作，"
                                "完成后点「停止并保存」"
                            )
                            tpl_cam_input = gr.Image(
                                label="摄像头",
                                sources=["webcam"],
                                streaming=True,
                                type="numpy",
                                height=240,
                            )
                            tpl_cam_display = gr.Image(
                                label="骨骼叠加",
                                type="numpy",
                                interactive=False,
                                height=240,
                            )
                            tpl_cam_status = gr.Textbox(
                                label="录制状态",
                                value="⏸️ 空闲",
                                interactive=False,
                            )
                            with gr.Row():
                                tpl_cam_start = gr.Button(
                                    "� 开始录制", variant="primary"
                                )
                                tpl_cam_stop = gr.Button(
                                    "⏹️ 停止并保存",
                                    variant="secondary",
                                    interactive=False,
                                )

            # ================== 模板骨骼演示（问题4） ==================
            gr.Markdown("---")
            gr.Markdown("#### 🦴 模板骨骼演示视频")
            gr.Markdown(
                "> 每个录入的模板都会自动生成骨骼演示视频。"
                "直接选择动作类别和模板名称即可播放对应的骨骼动画，"
                "风格与视频分析中【骨骼对比视频】的模板半区一致。"
            )

            with gr.Row():
                _init_prev_actions = _list_actions_with_templates()
                _init_prev_action = _init_prev_actions[0] if _init_prev_actions else None
                _init_prev_tpls = (
                    pipeline.template_library.list_templates(_init_prev_action)
                    if _init_prev_action else []
                )
                _init_prev_tpl = _init_prev_tpls[0] if _init_prev_tpls else None
                _init_prev_video = (
                    pipeline.get_template_demo_path(
                        _init_prev_action, _init_prev_tpl)
                    if _init_prev_action and _init_prev_tpl else None
                )

                prev_action_dropdown = gr.Dropdown(
                    choices=_init_prev_actions,
                    value=_init_prev_action,
                    label="动作类别",
                )
                prev_template_dropdown = gr.Dropdown(
                    choices=_init_prev_tpls,
                    value=_init_prev_tpl,
                    label="模板名称",
                )
                prev_refresh_btn = gr.Button("🔄 刷新列表")

            prev_video = gr.Video(
                label="🎬 模板骨骼演示",
                value=_init_prev_video,
                height=400,
                autoplay=True,
            )

            # ================== 删除模板 ==================
            gr.Markdown("---")
            gr.Markdown("#### 🗑️ 删除模板")
            gr.Markdown(
                "> 选择动作类别和模板名称，点击「删除」即可移除。"
                "当某动作下的最后一个模板被删除时，该动作类别也会一并从下拉列表中移除，"
                "其他 Tab（视频分析 / 实时模式）会立即同步更新。"
            )

            with gr.Row():
                _init_del_actions = _list_actions_with_templates()
                _init_del_action = _init_del_actions[0] if _init_del_actions else None
                _init_del_tpls = (
                    pipeline.template_library.list_templates(_init_del_action)
                    if _init_del_action else []
                )

                del_action_dropdown = gr.Dropdown(
                    choices=_init_del_actions,
                    value=_init_del_action,
                    label="动作类别",
                )
                del_template_dropdown = gr.Dropdown(
                    choices=_init_del_tpls,
                    value=(_init_del_tpls[0] if _init_del_tpls else None),
                    label="模板名称",
                )
                del_refresh_btn = gr.Button("🔄 刷新")
                del_template_btn = gr.Button("🗑️ 删除模板", variant="stop")
                del_action_btn = gr.Button("💥 删除整个动作", variant="stop")

            del_status = gr.Textbox(label="删除状态", interactive=False)

            # -- 绑定事件 --
            refresh_btn.click(
                fn=on_refresh_templates, outputs=[template_info]
            )
            upload_btn.click(
                fn=on_record_template,
                inputs=[tpl_video, tpl_name_input],
                outputs=[upload_status, vid_action_dropdown, rt_action_dropdown],
            )

            # 模板预览绑定（点击即播放，无需"生成"按钮）
            prev_refresh_btn.click(
                fn=on_tpl_preview_refresh,
                outputs=[
                    prev_action_dropdown, prev_template_dropdown, prev_video,
                ],
            )
            prev_action_dropdown.change(
                fn=on_tpl_preview_action_change,
                inputs=[prev_action_dropdown],
                outputs=[prev_template_dropdown, prev_video],
            )
            prev_template_dropdown.change(
                fn=on_tpl_preview_template_change,
                inputs=[prev_action_dropdown, prev_template_dropdown],
                outputs=[prev_video],
            )

            del_refresh_btn.click(
                fn=on_del_refresh_actions,
                outputs=[del_action_dropdown, del_template_dropdown],
            )
            del_action_dropdown.change(
                fn=on_del_action_change,
                inputs=[del_action_dropdown],
                outputs=[del_template_dropdown],
            )
            del_template_btn.click(
                fn=on_delete_template,
                inputs=[del_action_dropdown, del_template_dropdown],
                outputs=[
                    del_status,
                    del_action_dropdown,
                    del_template_dropdown,
                    vid_action_dropdown,
                    rt_action_dropdown,
                ],
            )
            del_action_btn.click(
                fn=on_delete_action,
                inputs=[del_action_dropdown],
                outputs=[
                    del_status,
                    del_action_dropdown,
                    del_template_dropdown,
                    vid_action_dropdown,
                    rt_action_dropdown,
                ],
            )

            # 摄像头录制模板
            tpl_cam_input.stream(
                fn=on_tpl_cam_frame,
                inputs=[tpl_cam_input],
                outputs=[tpl_cam_display],
                stream_every=0.1,
                show_progress="hidden",
            )
            tpl_cam_start.click(
                fn=on_tpl_start_recording,
                outputs=[tpl_cam_status, tpl_cam_start, tpl_cam_stop],
            )
            tpl_cam_stop.click(
                fn=on_tpl_stop_recording,
                inputs=[tpl_name_input],
                outputs=[
                    tpl_cam_status, tpl_cam_start, tpl_cam_stop,
                    vid_action_dropdown, rt_action_dropdown,
                ],
            )

        # ============================================================
        # Tab 4: 系统说明
        # ============================================================
        with gr.Tab("ℹ️ 系统说明"):
            gr.Markdown(_build_system_description())

    return app


def _build_system_description() -> str:
    """构建系统说明 Markdown"""
    return """
## 🏋️ 基于深度学习的人体动作矫正系统

### 📖 系统简介

本系统是一个面向健身/康复训练场景的智能动作矫正平台，能够通过计算机视觉和深度学习技术，
自动分析用户的运动姿态，并与标准动作模板进行对比，生成详细的矫正建议。

### 🔧 核心功能

| 功能 | 说明 |
|------|------|
| 📹 视频分析 | 上传视频或摄像头录制，获取完整矫正报告 |
| 📷 实时模式 | 摄像头实时跟做，边做边获得即时矫正提示 |
| 📋 模板管理 | 查看和管理标准动作模板库 |
| 🔧 算法切换 | DTW / FastDTW / DDTW / 自动选择 |

### 🏗️ 技术架构

```
视频/摄像头输入
      ↓
┌─────────────────────────┐
│  MediaPipe 姿态估计       │  ← 提取 33 个人体骨骼关键点
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  深度学习动作分类          │  ← ST-GCN / BiLSTM / Transformer
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  DTW 动态时间规整对比      │  ← 支持经典DTW / FastDTW / DDTW
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  规则引擎矫正反馈生成      │  ← 40+ 条专家规则
└─────────────────────────┘
      ↓
   矫正报告 + 可视化
```

### 📋 使用指南

#### 视频分析
1. 切换到 **📹 视频分析** Tab
2. 选择输入方式（上传视频 / 摄像头录制）
3. 选择动作类型和 DTW 算法
4. 上传视频或录制后，点击 **开始分析**
5. 查看矫正报告和偏差图

#### 实时模式（推荐体验）
1. 切换到 **📷 实时模式** Tab
2. 打开摄像头
3. 选择动作类型和算法
4. 点击 **开始跟做** → 3 秒倒计时 → 开始做动作
5. 右侧面板实时显示矫正建议
6. 完成后点击 **结束** → 查看总体评分和报告

#### DTW 算法对比
系统支持四种 DTW 模式，方便对比测试：
- **经典 DTW**: 精确但较慢，适合基准测试
- **FastDTW**: 近似线性复杂度，速度最快
- **DerivativeDTW**: 关注动作形状趋势，对时间偏移鲁棒
- **自动选择**: 实时用 FastDTW，离线用 DDTW（推荐）

### 🔬 支持的动作类型

| 动作 | 标识名 | 规则数 |
|------|--------|--------|
| 深蹲 | squat | 10+ |
| 手臂举起 | arm_raise | 10+ |
| 侧弯 | side_bend | 8+ |
| 弓步 | lunge | 10+ |
| 站立拉伸 | standing_stretch | 8+ |

### ⚠️ 注意事项

- **实时模式**建议手动选择动作类型，不要用"自动识别"
- 首次使用请运行 `python scripts/prepare_demo.py` 生成演示模板
- 实时反馈延迟目标 ≤300ms（MediaPipe ~50ms + DTW ~10ms + 网络 ~100ms）

### 📚 技术栈

- **姿态估计**: MediaPipe PoseLandmarker
- **深度学习**: PyTorch (ST-GCN, BiLSTM-Attention, Transformer)
- **动作对比**: DTW / FastDTW / DerivativeDTW
- **Web 框架**: Gradio
- **语言**: Python 3.9+

---
*华中科技大学 · 计算机科学与技术学院 · CS2201 张玉倍 · U202215369*
"""
