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

import cv2
import gradio as gr
import numpy as np

from .pipeline import AppPipeline, ACTION_DISPLAY_NAMES
from .session import SessionManager

logger = logging.getLogger(__name__)

# 实时模式性能参数
_PROCESS_WIDTH = 320       # 推理用缩放宽度（越小越快）
_SKIP_FRAMES = 0           # 帧跳过数（0 表示每帧都推理）
_frame_counter = 0         # 全局帧计数器
_last_annotated = None     # 上一帧骨骼叠加结果缓存


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

    def on_camera_frame(frame):
        """
        摄像头流处理回调

        关键修复：
        - 输入是摄像头原始帧，输出到独立的显示组件（不回写摄像头组件）
        - 可选帧跳过以提升帧率
        - 缩放图像再推理以降低延迟
        """
        global _frame_counter, _last_annotated

        if frame is None:
            return None

        _frame_counter += 1

        # 帧跳过：如果不是推理帧，直接用缓存的骨骼或原图
        if _SKIP_FRAMES > 0 and (_frame_counter % (_SKIP_FRAMES + 1)) != 0:
            if _last_annotated is not None:
                return _last_annotated
            return frame

        # 缩放图像以加快推理
        h, w = frame.shape[:2]
        if w > _PROCESS_WIDTH:
            scale = _PROCESS_WIDTH / w
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
            scale = 1.0

        # 姿态估计（在缩小的图像上进行）
        processed = pipeline.process_camera_frame(small)

        # 如果正在录制，缓存 landmarks
        if session.is_recording and processed.has_pose:
            session.add_frame(processed.landmarks)

        # 将骨骼绘制回原尺寸图像
        if processed.has_pose and processed.annotated_image is not None:
            if scale != 1.0:
                annotated = cv2.resize(
                    processed.annotated_image, (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                annotated = processed.annotated_image
        else:
            annotated = frame.copy()

        _last_annotated = annotated
        return annotated

    def on_start_recording():
        """开始录制"""
        session.start_recording()
        return (
            "🔴 录制中… 请执行动作，完成后点击「停止并分析」",
            gr.update(interactive=False),  # 开始按钮
            gr.update(interactive=True),   # 停止按钮
        )

    def on_stop_recording(action_choice):
        """停止录制并分析"""
        sequence = session.stop_recording()

        if sequence is None:
            session.finish_analysis()
            return (
                "⚠️ 录制帧数不足（至少 5 帧），请确保摄像头能看到全身，然后重试",
                "", None,
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        action_name = None if action_choice == "自动识别" else action_choice

        # 如果没有 checkpoint 且选了自动识别，自动 fallback 到第一个可用动作
        if action_name is None:
            actions = session.get_action_list()
            if actions:
                action_name = actions[0]
                logger.info(
                    "无分类模型，自动使用第一个动作类型: %s", action_name
                )
            else:
                session.finish_analysis()
                return (
                    "⚠️ 模板库为空且未选择动作类型，请先在「模板管理」中录入模板，"
                    "或手动选择动作类型后重试",
                    "", None,
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                )

        result = pipeline.analyze_sequence(
            sequence_data=sequence,
            action_name=action_name,
        )

        session.finish_analysis(result)

        return (
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
            gr.Markdown(
                "### 摄像头实时捕捉，录制动作后自动分析\n"
                "**操作步骤**: 点击摄像头区域开启 → 选择动作类型 → "
                "点击「开始录制」→ 做动作 → 点击「停止并分析」"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # ---- 摄像头输入 + 骨骼叠加显示 ----
                    # 使用 streaming=True 获取实时帧流
                    # 输出到独立的 camera_display 组件，避免回写导致闪烁
                    camera_input = gr.Image(
                        label="摄像头（点击开启）",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=240,      # 缩小输入预览
                    )

                    # 骨骼叠加结果（独立显示组件）
                    camera_display = gr.Image(
                        label="骨骼叠加画面",
                        type="numpy",
                        interactive=False,
                        height=360,
                    )

                    camera_status = gr.Textbox(
                        label="状态",
                        value="⏸️ 空闲 — 请先打开摄像头",
                        interactive=False,
                    )
                    cam_action_dropdown = gr.Dropdown(
                        choices=session.get_action_choices(),
                        value=(
                            session.get_action_choices()[1]
                            if len(session.get_action_choices()) > 1
                            else "自动识别"
                        ),
                        label="动作类型（建议手动选择）",
                        info="实时模式建议手动选择动作类型",
                    )

                    with gr.Row():
                        start_btn = gr.Button(
                            "🔴 开始录制", variant="primary"
                        )
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

            # 摄像头流处理 — 输入从 camera_input，输出到 camera_display
            # 不回写 camera_input，避免自循环导致闪烁
            camera_input.stream(
                fn=on_camera_frame,
                inputs=[camera_input],
                outputs=[camera_display],
            )

            # 录制控制
            start_btn.click(
                fn=on_start_recording,
                outputs=[camera_status, start_btn, stop_btn],
            )

            stop_btn.click(
                fn=on_stop_recording,
                inputs=[cam_action_dropdown],
                outputs=[
                    camera_status,
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
                    template_info_display = gr.Markdown(
                        '点击"刷新"查看模板列表'
                    )
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
                    record_btn = gr.Button(
                        "📥 录入模板", variant="primary"
                    )
                    record_status = gr.Textbox(
                        label="录入状态",
                        interactive=False,
                    )

            refresh_btn.click(
                fn=on_refresh_templates,
                outputs=[template_info_display],
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

#### 视频分析（推荐）
1. 切换到 **📹 视频分析** Tab
2. 上传一段运动视频（mp4/avi/mov）
3. 选择动作类型（如 squat）
4. 点击 **开始分析**
5. 等待分析完成，查看矫正报告和偏差图

#### 实时模式
1. 切换到 **📷 实时模式** Tab
2. 点击摄像头区域打开摄像头
3. **选择动作类型**（建议手动选择，不要用自动识别）
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

### ⚠️ 注意事项

- **实时模式**建议手动选择动作类型（下拉框选择具体动作），不要使用"自动识别"
- 如果分析失败，请先确认模板管理中有对应动作的模板数据
- 首次使用请运行 `python scripts/prepare_demo.py` 生成演示模板

### 📚 技术栈

- **姿态估计**: MediaPipe PoseLandmarker
- **深度学习**: PyTorch (ST-GCN, BiLSTM-Attention, Transformer)
- **动作对比**: DTW (动态时间规整)
- **Web 框架**: Gradio
- **语言**: Python 3.9+

---
*华中科技大学 · 计算机科学与技术学院 · CS2201 张玉倍 · U202215369*
"""
