"""
Gradio Web 交互界面

基于 Gradio Blocks 的多 Tab 布局：
- Tab 1: 📹 视频分析 — 上传视频 → 矫正报告
- Tab 2: 📷 实时模式 — 摄像头实时骨骼叠加 + 录制分析
- Tab 3: 📋 模板管理 — 查看/录入标准动作模板
- Tab 4: ℹ️ 系统说明 — 使用指南和技术架构
"""

import logging
import time
from typing import Optional

import gradio as gr
import numpy as np

from .pipeline import AppPipeline, ACTION_DISPLAY_NAMES
from .session import SessionManager

logger = logging.getLogger(__name__)


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
    # 回调函数
    # ================================================================

    def on_analyze_video(video, action_choice, progress=gr.Progress()):
        """视频分析回调"""
        if video is None:
            return "⚠️ 请先上传视频文件", None, None

        # 解析动作选择
        action_name = None if action_choice == "自动识别" else action_choice

        # 进度回调
        def progress_cb(step, total, msg):
            progress(step / total, desc=msg)

        progress(0, desc="开始分析…")
        result = pipeline.analyze_video(
            video_path=video,
            action_name=action_name,
            progress_callback=progress_cb,
        )

        # 缓存结果
        session.last_result = result

        return (
            result.report_text,
            result.deviation_plot_path,
            result.summary(),
        )

    def on_camera_frame(frame, is_recording):
        """摄像头帧处理回调"""
        if frame is None:
            return None, f"帧缓存: 0"

        processed = pipeline.process_camera_frame(frame)

        # 如果在录制中，缓存 landmarks
        if is_recording and processed.has_pose:
            session.add_frame(processed.landmarks)

        status = f"帧缓存: {session.frame_count}"
        if is_recording:
            status = f"🔴 录制中 | {status}"

        return processed.annotated_image, status

    def on_start_recording():
        """开始录制"""
        session.start_recording()
        return (
            True,   # is_recording state
            "🔴 录制中… 请执行动作",
            gr.update(interactive=False),  # 开始按钮
            gr.update(interactive=True),   # 停止按钮
        )

    def on_stop_recording(action_choice):
        """停止录制并分析"""
        sequence = session.stop_recording()

        if sequence is None:
            session.finish_analysis()
            return (
                False,
                "⚠️ 录制帧数不足（至少 5 帧），请重试",
                "", None,
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        action_name = None if action_choice == "自动识别" else action_choice
        result = pipeline.analyze_sequence(
            sequence_data=sequence,
            action_name=action_name,
        )

        session.finish_analysis(result)

        return (
            False,
            "✅ 分析完成",
            result.report_text,
            result.deviation_plot_path,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    def on_refresh_templates():
        """刷新模板列表"""
        session.refresh_action_list()
        info = pipeline.get_template_info()

        if not info:
            return "📭 模板库为空，请录入标准动作模板。", gr.update(choices=session.get_action_choices())

        lines = ["## 📋 模板库概览\n"]
        for item in info:
            lines.append(
                f"- **{item['display_name']}** (`{item['action']}`): "
                f"{item['count']} 个模板"
            )
            for t in item['templates']:
                lines.append(f"  - {t}")

        return "\n".join(lines), gr.update(choices=session.get_action_choices())

    def on_record_template(video, action_input):
        """录入新模板"""
        if video is None:
            return "⚠️ 请上传视频文件"
        if not action_input or not action_input.strip():
            return "⚠️ 请输入动作名称"

        action_name = action_input.strip()
        success = pipeline.record_template(video, action_name)

        if success:
            session.refresh_action_list()
            return f"✅ 模板录入成功: {action_name}"
        else:
            return "❌ 模板录入失败，请检查视频是否包含有效的人体姿态"

    # ================================================================
    # 界面构建
    # ================================================================

    with gr.Blocks(
        title="人体动作矫正系统",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .sub-title { text-align: center; color: #666; margin-bottom: 1.5em; }
        """
    ) as app:

        # 标题
        gr.Markdown(
            "# 🏋️ 基于深度学习的人体动作矫正系统",
            elem_classes=["main-title"],
        )
        gr.Markdown(
            "*CS2201 张玉倍 — 本科毕业设计*",
            elem_classes=["sub-title"],
        )

        # ============================================================
        # Tab 1: 视频分析
        # ============================================================
        with gr.Tab("📹 视频分析"):
            gr.Markdown("### 上传视频，获取动作矫正报告")

            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="上传视频",
                        sources=["upload"],
                    )
                    action_dropdown = gr.Dropdown(
                        choices=session.get_action_choices(),
                        value="自动识别",
                        label="动作类型",
                        info="选择动作类型，或使用自动识别",
                    )
                    analyze_btn = gr.Button(
                        "🔍 开始分析",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=2):
                    summary_output = gr.Textbox(
                        label="📊 分析摘要",
                        lines=5,
                        interactive=False,
                    )
                    report_output = gr.Textbox(
                        label="📝 矫正报告",
                        lines=15,
                        interactive=False,
                    )
                    plot_output = gr.Image(
                        label="📈 偏差分析图",
                        type="filepath",
                    )

            analyze_btn.click(
                fn=on_analyze_video,
                inputs=[video_input, action_dropdown],
                outputs=[report_output, plot_output, summary_output],
            )

        # ============================================================
        # Tab 2: 实时模式
        # ============================================================
        with gr.Tab("📷 实时模式"):
            gr.Markdown("### 摄像头实时捕捉，录制动作后自动分析")

            is_recording_state = gr.State(False)

            with gr.Row():
                with gr.Column(scale=1):
                    camera_input = gr.Image(
                        label="摄像头画面",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                    )
                    camera_status = gr.Textbox(
                        label="状态",
                        value="⏸️ 空闲",
                        interactive=False,
                    )
                    cam_action_dropdown = gr.Dropdown(
                        choices=session.get_action_choices(),
                        value="自动识别",
                        label="动作类型",
                    )

                    with gr.Row():
                        start_btn = gr.Button("🔴 开始录制", variant="primary")
                        stop_btn = gr.Button(
                            "⏹️ 停止并分析",
                            variant="secondary",
                            interactive=False,
                        )

                with gr.Column(scale=1):
                    cam_report_output = gr.Textbox(
                        label="📝 矫正报告",
                        lines=15,
                        interactive=False,
                    )
                    cam_plot_output = gr.Image(
                        label="📈 偏差分析图",
                        type="filepath",
                    )

            # 摄像头流处理
            camera_input.stream(
                fn=on_camera_frame,
                inputs=[camera_input, is_recording_state],
                outputs=[camera_input, camera_status],
            )

            # 录制控制
            start_btn.click(
                fn=on_start_recording,
                outputs=[is_recording_state, camera_status, start_btn, stop_btn],
            )

            stop_btn.click(
                fn=on_stop_recording,
                inputs=[cam_action_dropdown],
                outputs=[
                    is_recording_state, camera_status,
                    cam_report_output, cam_plot_output,
                    start_btn, stop_btn,
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
                    template_info_display = gr.Markdown('点击"刷新"查看模板列表')
                    refresh_btn = gr.Button("🔄 刷新列表")

                with gr.Column(scale=1):
                    gr.Markdown("#### ➕ 录入新模板")
                    template_video_input = gr.Video(
                        label="上传标准动作视频",
                        sources=["upload"],
                    )
                    template_action_input = gr.Textbox(
                        label="动作名称",
                        placeholder="例如: squat, arm_raise, lunge ...",
                        info="英文标识名，新动作会自动创建类别",
                    )
                    record_btn = gr.Button("📥 录入模板", variant="primary")
                    record_status = gr.Textbox(
                        label="录入状态",
                        interactive=False,
                    )

            # 隐藏的 Dropdown 用于同步更新（实际作用是触发状态刷新）
            hidden_dropdown_update = gr.Dropdown(visible=False)

            refresh_btn.click(
                fn=on_refresh_templates,
                outputs=[template_info_display, hidden_dropdown_update],
            )

            record_btn.click(
                fn=on_record_template,
                inputs=[template_video_input, template_action_input],
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
| 📹 视频分析 | 上传运动视频，获取完整的动作矫正报告 |
| 📷 实时模式 | 摄像头实时捕捉，录制动作后即时分析 |
| 📋 模板管理 | 查看和管理标准动作模板库 |

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
│  DTW 动态时间规整对比      │  ← 与标准模板逐关节对比
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  规则引擎矫正反馈生成      │  ← 50+ 条专家规则，中文自然语言建议
└─────────────────────────┘
      ↓
   矫正报告 + 可视化
```

### 📋 使用指南

#### 视频分析
1. 切换到 **📹 视频分析** Tab
2. 上传一段运动视频（mp4/avi/mov）
3. 选择动作类型（或选"自动识别"）
4. 点击 **开始分析**
5. 等待分析完成，查看矫正报告和偏差图

#### 实时模式
1. 切换到 **📷 实时模式** Tab
2. 允许浏览器访问摄像头
3. 确认画面中显示了骨骼标注
4. 点击 **开始录制**，执行动作
5. 完成后点击 **停止并分析**
6. 查看矫正报告

#### 模板管理
1. 切换到 **📋 模板管理** Tab
2. 点击 **刷新列表** 查看已有模板
3. 要录入新模板：上传标准动作视频 → 输入动作名称 → 点击 **录入模板**

### 🔬 支持的动作类型

| 动作 | 标识名 | 规则数 |
|------|--------|--------|
| 深蹲 | squat | 10+ |
| 手臂举起 | arm_raise | 10+ |
| 侧弯 | side_bend | 8+ |
| 弓步 | lunge | 10+ |
| 站立拉伸 | standing_stretch | 8+ |

### 📚 技术栈

- **姿态估计**: MediaPipe PoseLandmarker
- **深度学习**: PyTorch (ST-GCN, BiLSTM-Attention, Transformer)
- **动作对比**: DTW (动态时间规整)
- **Web 框架**: Gradio
- **语言**: Python 3.9+

---
*华中科技大学 · 计算机科学与技术学院 · CS2201 张玉倍 · U202215369*
"""
