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
_PROCESS_WIDTH = 480   # 推理用缩放宽度

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
        """缩放帧用于推理"""
        if frame is None:
            return frame, 1.0
        h, w = frame.shape[:2]
        if w > _PROCESS_WIDTH:
            scale = _PROCESS_WIDTH / w
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            return small, scale
        return frame, 1.0

    # ================================================================
    # Tab 1 回调：视频分析
    # ================================================================

    def on_analyze_video(video, action_choice, algo_label, progress=gr.Progress()):
        """上传视频分析"""
        if video is None:
            return "⚠️ 请先上传视频文件", None, None

        action_name = _resolve_action(action_choice)
        algo_id = _resolve_algorithm(algo_label)
        pipeline.set_algorithm(algo_id)

        def progress_cb(step, total, msg):
            progress(step / total, desc=msg)

        progress(0, desc="开始分析…")
        result = pipeline.analyze_video(
            video_path=video,
            action_name=action_name,
            progress_callback=progress_cb,
        )
        session.last_result = result
        return result.report_text, result.deviation_plot_path, result.summary()

    def on_start_cam_recording():
        """视频分析 Tab — 摄像头录制：开始"""
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
        processed = pipeline.process_camera_frame(frame)
        if session.is_recording and processed.has_pose:
            session.add_frame(processed.landmarks)
        return processed.annotated_image

    def on_stop_cam_recording(action_choice, algo_label):
        """视频分析 Tab — 摄像头录制：停止并分析"""
        sequence = session.stop_recording()
        if sequence is None:
            session.finish_analysis()
            return (
                "⚠️ 录制帧数不足（至少 5 帧），请确保全身在画面中",
                "", None,
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

        # 开始倒计时
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
        """
        if frame is None:
            return None, gr.update(), gr.update()

        # 缩放
        small, scale = _scale_frame(frame)
        h, w = frame.shape[:2]

        # ---- 倒计时状态 ----
        if session.is_countdown:
            remaining = session.countdown_remaining
            if session.countdown_finished:
                session.start_realtime()
                status = "🟢 开始跟做！"
                feedback_md = "**实时分析已启动，请开始做动作…**"
            else:
                num = math.ceil(remaining)
                status = f"⏱️ {num}..."
                feedback_md = f"**倒计时: {num}** — 请准备好姿势"

            # 骨骼叠加（倒计时期间也显示骨骼）
            processed = pipeline.process_camera_frame(small)
            if processed.annotated_image is not None and scale != 1.0:
                display = cv2.resize(processed.annotated_image, (w, h))
            else:
                display = processed.annotated_image if processed.annotated_image is not None else frame

            return display, status, feedback_md

        # ---- 实时分析状态 ----
        if session.is_realtime:
            frame_idx = session.advance_realtime_frame()
            expected_total = 90  # 约 3 秒 @30fps

            processed, snapshot = pipeline.process_realtime_frame(
                small, frame_idx, expected_total
            )

            # 缩放回原尺寸
            if processed.annotated_image is not None and scale != 1.0:
                display = cv2.resize(processed.annotated_image, (w, h))
            else:
                display = processed.annotated_image if processed.annotated_image is not None else frame

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
        if processed.annotated_image is not None and scale != 1.0:
            display = cv2.resize(processed.annotated_image, (w, h))
        else:
            display = processed.annotated_image if processed.annotated_image is not None else frame

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

    def on_record_template(video, action_input):
        if video is None:
            return "⚠️ 请上传视频文件"
        if not action_input or not action_input.strip():
            return "⚠️ 请输入动作名称"
        action_name = action_input.strip()
        success = pipeline.record_template(video, action_name)
        if success:
            session.refresh_action_list()
            return f"✅ 模板录入成功: {action_name}"
        return "❌ 模板录入失败，请检查视频是否包含有效的人体姿态"

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
                        video_input = gr.Video(
                            label="上传视频", sources=["upload"]
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

            # 上传视频分析
            analyze_btn.click(
                fn=on_analyze_video,
                inputs=[video_input, vid_action_dropdown, vid_algo_dropdown],
                outputs=[vid_report, vid_plot, vid_summary],
            )

            # 摄像头录制
            cam_rec_input.stream(
                fn=on_cam_record_frame,
                inputs=[cam_rec_input],
                outputs=[cam_rec_display],
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
                    cam_rec_report, cam_rec_plot,
                    cam_rec_start, cam_rec_stop,
                ],
            )

        # ============================================================
        # Tab 2: 实时模式
        # ============================================================
        with gr.Tab("📷 实时模式"):
            gr.Markdown(
                "### 实时跟做训练，边做边获得矫正提示\n"
                "**流程**: 选择动作和算法 → 点击「开始跟做」→ "
                "3 秒倒计时 → 跟做动作（右侧实时显示建议）→ "
                "点击「结束」→ 查看总体报告"
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
                with gr.Column(scale=1):
                    gr.Markdown("#### 📂 模板列表")
                    template_info = gr.Markdown('点击"刷新"查看模板列表')
                    refresh_btn = gr.Button("🔄 刷新列表")
                with gr.Column(scale=1):
                    gr.Markdown("#### ➕ 录入新模板")
                    tpl_video = gr.Video(
                        label="上传标准动作视频", sources=["upload"]
                    )
                    tpl_name = gr.Textbox(
                        label="动作名称",
                        placeholder="例如: squat, arm_raise, lunge",
                        info="英文标识名",
                    )
                    record_btn = gr.Button("📥 录入模板", variant="primary")
                    record_status = gr.Textbox(
                        label="录入状态", interactive=False
                    )

            refresh_btn.click(
                fn=on_refresh_templates, outputs=[template_info]
            )
            record_btn.click(
                fn=on_record_template,
                inputs=[tpl_video, tpl_name],
                outputs=[record_status],
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
