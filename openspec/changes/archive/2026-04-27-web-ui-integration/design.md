## Context

系统核心算法模块已全部完成：

| 模块 | 位置 | 功能 |
|------|------|------|
| 姿态估计 | `src/pose_estimation/` | MediaPipe 33 关键点提取 |
| 数据预处理 | `src/data/` | 标准化、增强、模板库管理 |
| 动作对比 | `src/action_comparison/` | DTW 相似度 + 关节偏差分析 |
| 深度学习 | `src/models/` | ST-GCN / BiLSTM / Transformer 分类器 |
| 矫正反馈 | `src/correction/` | 规则引擎 + 自然语言报告 |

当前状态：所有模块仅通过 `scripts/` 下的 Python 脚本调用，缺乏统一 Web 界面。`src/app/` 目录已预留但为空。

约束：
- Gradio 4.x 作为 UI 框架（Python 一体化，零前端代码）
- 需支持摄像头实时流和视频文件上传两种输入
- 界面需美观、易操作，适合答辩现场演示

### 数据流图

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Gradio UI  │────▶│  AppPipeline     │────▶│  结果展示         │
│  (Tab 布局)  │     │  (端到端编排)     │     │  (报告+可视化)    │
└──────────────┘     └──────────────────┘     └─────────────────┘
     │                      │
     ├─ 视频上传             ├─ PoseEstimator      → 骨骼序列
     ├─ 摄像头流             ├─ ActionPredictor    → 动作分类
     └─ 动作选择             ├─ ActionComparator   → DTW 对比
                             ├─ CorrectionPipeline → 矫正报告
                             └─ ReportVisualizer   → 可视化图
```

## Goals / Non-Goals

**Goals:**
- 提供一键启动的 Web 演示界面，答辩时可直接使用
- 支持视频上传分析（主功能）：上传 → 骨骼提取 → 分类 → 对比 → 矫正报告
- 支持实时摄像头模式：实时骨骼叠加 + 录制完成后自动分析
- 提供模板管理页面：查看已有模板、从视频录入新模板
- 界面简洁美观，中文标注，支持深色/浅色主题

**Non-Goals:**
- 不做多用户并发（单机演示场景）
- 不做用户认证和权限管理
- 不做移动端适配
- 不做模型在线训练（仅推理）

## Decisions

### 1. UI 框架：Gradio 4.x

**选择理由**：
- 纯 Python，无需前端代码，开发效率最高
- 内置 `gr.Video`、`gr.Image`、`gr.Plot` 等组件，天然适合 CV 项目
- 支持 `gr.Blocks` 自定义布局和 `gr.Tab` 多页签
- 一行 `demo.launch()` 即可启动，适合答辩演示

**备选方案**：Streamlit（交互模型不如 Gradio 灵活）、Flask+HTML（需要写前端）

### 2. 应用层架构：AppPipeline 编排模式

在 `src/app/pipeline.py` 中实现 `AppPipeline` 类，作为所有 UI 回调的唯一入口：

```python
class AppPipeline:
    def analyze_video(self, video_path, action_name=None) -> AnalysisResult
    def analyze_sequence(self, sequence, action_name=None) -> AnalysisResult
    def process_camera_frame(self, frame) -> ProcessedFrame
    def record_template(self, video_path, action_name) -> bool
```

**输入格式**：
- `video_path`: `str`，视频文件路径
- `sequence`: `np.ndarray`，shape `(T, 33, 4)`
- `frame`: `np.ndarray`，shape `(H, W, 3)` BGR

**输出格式**：
- `AnalysisResult`: dataclass 包含 `report: CorrectionReport`, `skeleton_video_path: str`, `deviation_plot_path: str`, `summary_text: str`

### 3. Tab 布局设计

| Tab | 组件 | 功能 |
|-----|------|------|
| 📹 视频分析 | Video + Dropdown + Button → Report + Plot + Video | 上传视频，选择动作类型（或自动识别），生成完整报告 |
| 📷 实时模式 | Image(streaming) + Button → Image + Report | 摄像头实时骨骼叠加，手动开始/停止，停止后自动分析 |
| 📋 模板管理 | Dropdown + DataTable + Video | 查看已有模板、录入新模板 |
| ℹ️ 系统说明 | Markdown | 系统介绍、使用指南、技术架构 |

### 4. 实时摄像头流处理

使用 Gradio 的 `gr.Image(sources=["webcam"], streaming=True)` 组件：
- 每帧经过 MediaPipe 提取关键点并叠加骨骼
- 点击"开始录制"后缓存帧序列到内存
- 点击"停止并分析"后对缓存序列执行完整矫正流水线
- 使用 `gr.State` 管理录制状态和帧缓存

### 5. 进度反馈

视频分析为耗时操作（10-30 秒），使用 Gradio 的 `gr.Progress` 提供步骤提示：
- "正在提取骨骼关键点… (1/4)"
- "正在分析动作类型… (2/4)"
- "正在对比标准动作… (3/4)"
- "正在生成矫正报告… (4/4)"

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|---------|
| 摄像头实时流在 Gradio 中帧率受限（~10-15fps） | 仅做骨骼叠加预览，不实时计算 DTW |
| 长视频处理时间过长 | 限制最大帧数（300 帧 ≈ 10 秒），超长视频自动采样 |
| MediaPipe GPU 与 PyTorch GPU 内存竞争 | MediaPipe 默认用 CPU，PyTorch 模型较小（<50MB） |
| Gradio 版本兼容性 | 锁定 `gradio>=4.0,<5.0` |

## Open Questions

- 无（技术方案已明确）
