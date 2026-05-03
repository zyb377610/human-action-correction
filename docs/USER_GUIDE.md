# 基于深度学习的人体动作矫正系统 — 使用文档

> **毕业设计** | CS2201 | 张玉倍 | U202215369  
> 文档版本: v1.0 | 更新日期: 2026-05-02

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境准备](#2-环境准备)
3. [快速开始](#3-快速开始)
4. [功能模块详解](#4-功能模块详解)
   - 4.1 视频分析模式
   - 4.2 实时摄像头模式
   - 4.3 模板管理
   - 4.4 系统说明
5. [命令行工具](#5-命令行工具)
6. [API 参考](#6-api-参考)
7. [项目架构](#7-项目架构)
8. [常见问题](#8-常见问题)

---

## 1. 系统概述

本系统是一款基于 **MediaPipe 姿态估计** 与 **DTW 动态时间规整算法** 的人体动作矫正工具。系统通过摄像头或视频输入，精准提取 **33 个人体骨骼关键点**（含 17 个核心运动关节点），与标准动作模板库中的动作进行骨骼序列对比，生成 **0–100 分量化评分**，并以视觉高亮、文字提示等形式提供实时矫正反馈。

### 1.1 核心功能

| 功能 | 说明 |
|------|------|
| 🎯 姿态估计 | MediaPipe PoseLandmarker，提取 33 个关键点，准确率 ≥ 85% |
| 📏 动作对比 | 经典 DTW / FastDTW / Derivative DTW，支持 3 种距离度量 |
| 📊 量化评分 | 0–100 分制，结合相似度与关节偏差综合计算 |
| 📹 离线视频 | 上传 MP4/AVI/MOV 视频，自动姿态提取 + 矫正分析 |
| 📷 实时摄像头 | 实时骨骼叠加 + 逐帧矫正建议，延迟 ≤ 300ms |
| 🗣️ 矫正反馈 | 50+ 条中文自然语言矫正规则，覆盖 5 种标准动作 |
| 📋 模板管理 | 支持查看、录入、自定义上传标准动作模板 |
| 📈 数据可视化 | 偏差柱状图、骨骼对比图、对齐路径图 |
| 🌐 Web 界面 | 基于 Gradio 的 4-Tab 交互界面，≤3 步操作 |

### 1.2 支持的动作类型

| 动作标识 | 中文名称 | 描述 |
|----------|----------|------|
| `squat` | 深蹲 | 双脚与肩同宽，臀部下沉至大腿与地面平行 |
| `arm_raise` | 举臂 | 双臂从体侧抬起至过头顶伸直 |
| `side_bend` | 侧弯 | 单手叉腰，另一侧手臂引导躯干侧向弯曲 |
| `lunge` | 弓步 | 前腿屈膝90度，后腿膝盖接近地面 |
| `standing_stretch` | 站立拉伸 | 双臂上举，身体向上延伸 |

### 1.3 技术栈

| 技术方向 | 关键技术 |
|----------|----------|
| 姿态估计 | MediaPipe PoseLandmarker (Tasks API) |
| 动作对比 | DTW / FastDTW / Derivative DTW |
| 距离度量 | 欧氏距离 / 余弦距离 / 曼哈顿距离 |
| 深度学习 | ST-GCN / BiLSTM-Attention / Transformer |
| 矫正引擎 | 规则引擎 (50+ 条专家规则) |
| 数据预处理 | 缺失插值、Savitzky-Golay 平滑、帧数归一化 |
| Web 框架 | Gradio 4.x |
| 可视化 | Matplotlib + OpenCV |

---

## 2. 环境准备

### 2.1 系统要求

- **操作系统**: Windows 10/11、macOS 12+、Ubuntu 20.04+
- **Python**: 3.9 或以上
- **CUDA**: 11.8+（推荐，用于 GPU 加速；CPU 模式也可运行）
- **摄像头**: 内置或 USB 摄像头（实时模式需要）

### 2.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/zyb377610/human-action-correction.git
cd human-action-correction

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 2.3 模型文件

MediaPipe 姿态估计模型文件需放置在 `models/` 目录下：

| 文件 | 用途 | 下载地址 |
|------|------|----------|
| `pose_landmarker_full.task` | 标准模型（推荐） | [Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) |
| `pose_landmarker_heavy.task` | 高精度模型 | 同上 |
| `pose_landmarker_lite.task` | 轻量模型 | 同上 |

> 项目已自带 `pose_landmarker_full.task`。

---

## 3. 快速开始

### 3.1 三步启动

```bash
# Step 1: 生成演示模板数据（5种标准动作的合成模板）
python scripts/prepare_demo.py

# Step 2: 启动 Web 界面
python scripts/launch_app.py

# Step 3: 在浏览器中访问
# http://localhost:7860
```

### 3.2 自定义配置启动

```bash
# 指定端口
python scripts/launch_app.py --port 8080

# 指定模板目录
python scripts/launch_app.py --templates data/templates

# 生成公网分享链接（可远程访问）
python scripts/launch_app.py --share

# 启用调试模式
python scripts/launch_app.py --debug

# 加载分类模型（支持自动动作识别）
python scripts/launch_app.py --checkpoint outputs/checkpoints/best_model.pth --model-type bilstm
```

---

## 4. 功能模块详解

### 4.1 📹 视频分析 (Tab 1)

**用途**: 上传本地视频文件或通过摄像头录制动作，获取完整的矫正分析报告。

**操作步骤** (≤3步):
1. 选择动作类型（或选择"自动识别"）
2. 上传视频文件，或点击"开始录制"执行动作后"停止录制"
3. 查看矫正报告（评分 + 文字建议 + 偏差柱状图）

> 💡 **支持的视频格式**: MP4、AVI、MOV、WebM、MKV 等。
> 如 AVI 文件在界面中无法预览，不影响分析——后端使用 OpenCV 处理。
> 如遇编码问题，建议用格式工厂 / HandBrake 转为 MP4 (H.264)。

**输出内容**:
- **质量评分**: 0–100 分
- **DTW 相似度**: 0–100%
- **矫正建议**: 按优先级排序（高/中/低）
- **偏差柱状图**: 各关节偏差可视化
- **🎬 骨骼对比视频**: 逐帧播放原始视频 + 骨骼叠加，关节颜色标注偏差大小（🟢正常/🟡轻微/🔴需矫正），右上角显示标准模板骨架，底部滚动矫正建议
- **整体评语**: 优秀/良好/需改进/偏差较大

### 4.2 📷 实时模式 (Tab 2)

**用途**: 打开摄像头，实时显示骨骼叠加画面，逐帧提供矫正建议。

**操作步骤**:
1. 选择动作类型
2. 点击"开始实时分析"
3. 摄像头画面中按标准动作执行，实时观察骨骼标注和矫正建议

**实时反馈内容**:
- 骨骼关键点 + 连接线叠加显示（绿色=正常，灰色=低可见度）
- 关节角度数值标注
- Markdown 格式实时建议面板（关节偏差、方向、严重等级）
- 窗口相似度指示

**性能指标**:
- 单帧处理延迟: ≤ 10ms（不含 MediaPipe 推理）
- 总延迟: ≤ 300ms
- 显示分辨率: 480px 宽（优化传输）

### 4.3 📋 模板管理 (Tab 3)

**用途**: 查看现有模板、录入新标准动作模板。

**功能**:
- 查看各动作类别的模板列表
- 通过上传视频或摄像头录制新的标准动作模板
- 模板存储在 `data/templates/<action_name>/` 目录
- 支持视频格式: MP4、AVI、MOV、WebM、MKV（推荐 MP4 H.264）

### 4.4 ℹ️ 系统说明 (Tab 4)

**用途**: 显示使用指南、技术架构图和系统信息。

---

## 5. 命令行工具

### 5.1 `launch_app.py` — 启动 Web 界面

```bash
python scripts/launch_app.py [选项]

选项:
  --templates PATH    模板库目录路径 (默认: data/templates)
  --checkpoint PATH   分类模型 checkpoint 路径
  --model-type TYPE   模型类型: stgcn/bilstm/transformer (默认: bilstm)
  --port PORT         Web 服务端口 (默认: 7860)
  --share             生成公网分享链接
  --debug             启用调试模式
```

### 5.2 `prepare_demo.py` — 生成演示数据

```bash
python scripts/prepare_demo.py [--templates PATH]
```

为 5 种动作（深蹲、举臂、侧弯、弓步、站立拉伸）各生成一个合成模板。

### 5.3 `batch_process.py` — 离线批处理

```bash
python scripts/batch_process.py --input videos/ --output results/
```

批量处理视频目录中的所有视频文件。

### 5.4 `train_model.py` — 训练分类模型

```bash
python scripts/train_model.py --config configs/default.yaml
```

### 5.5 `evaluate_model.py` — 评估模型

```bash
python scripts/evaluate_model.py --checkpoint outputs/checkpoints/best_model.pth
```

### 5.6 其他工具

| 脚本 | 用途 |
|------|------|
| `process_video.py` | 单视频姿态提取 |
| `compare_action.py` | 两个动作序列对比 |
| `record_action.py` | 录制动作序列 |
| `run_correction.py` | 运行矫正分析 |
| `demo_pose.py` | 姿态估计演示 |

---

## 6. API 参考

### 6.1 PoseEstimator（姿态估计）

```python
from src.pose_estimation.estimator import PoseEstimator

estimator = PoseEstimator(model_complexity=1)  # 0=Lite, 1=Full, 2=Heavy
frame = estimator.estimate_frame(image)         # 单帧估计
sequence = estimator.estimate_video(source)      # 视频流估计
```

### 6.2 ActionComparator（动作对比）

```python
from src.action_comparison.comparison import ActionComparator

comparator = ActionComparator(algorithm="dtw", metric="euclidean")
result = comparator.compare(user_sequence, template_sequence)
print(f"相似度: {result.similarity:.2%}")
```

### 6.3 CorrectionPipeline（矫正流水线）

```python
from src.correction.pipeline import CorrectionPipeline

pipeline = CorrectionPipeline(templates_dir="data/templates")
report = pipeline.analyze(user_sequence, action_name="squat")
print(report.to_text())
```

### 6.4 AppPipeline（应用层流水线）

```python
from src.app.pipeline import AppPipeline

pipeline = AppPipeline(templates_dir="data/templates")
result = pipeline.analyze_video("my_video.mp4", action_name="squat")
print(result.report_text)
```

---

## 7. 项目架构

```
human-action-correction/
├── configs/                        # 配置文件
│   ├── default_config.yaml         # 系统默认配置
│   └── default.yaml                # 完整配置（含模型超参）
├── src/
│   ├── pose_estimation/            # Phase 1: 姿态估计
│   │   ├── estimator.py            #   MediaPipe PoseLandmarker 封装
│   │   ├── data_types.py           #   PoseLandmark/PoseFrame/PoseSequence
│   │   ├── feature_extractor.py    #   关节角度/骨骼长度/速度特征
│   │   ├── visualizer.py           #   骨骼绘制（17核心关节）
│   │   └── video_source.py         #   视频/摄像头源抽象
│   ├── data/                       # Phase 2: 数据处理
│   │   ├── preprocessing.py        #   插值/平滑/帧数归一化
│   │   ├── template_library.py     #   标准动作模板库
│   │   └── augmentation.py         #   数据增强
│   ├── action_comparison/          # Phase 3: 动作对比
│   │   ├── dtw_algorithms.py       #   DTW/FastDTW/DDTW
│   │   ├── distance_metrics.py     #   欧氏/余弦/曼哈顿距离
│   │   ├── comparison.py           #   对比器（特征转换→DTW→相似度）
│   │   ├── deviation_analyzer.py   #   逐关节偏差分析
│   │   └── visualizer.py           #   对齐路径可视化
│   ├── models/                     # Phase 4: 深度学习
│   │   ├── stgcn.py                #   ST-GCN 时空图卷积
│   │   ├── bilstm.py               #   BiLSTM-Attention
│   │   ├── transformer_model.py    #   Transformer
│   │   ├── trainer.py              #   训练器
│   │   └── predictor.py            #   推理器
│   ├── correction/                 # Phase 5: 矫正反馈
│   │   ├── rules.py                #   规则引擎 (50+条规则)
│   │   ├── feedback.py             #   反馈报告生成
│   │   ├── pipeline.py             #   端到端矫正流水线
│   │   ├── angle_utils.py          #   角度计算工具
│   │   ├── realtime_feedback.py    #   实时反馈引擎
│   │   └── report_visualizer.py    #   报告可视化
│   ├── app/                        # Phase 6: Web 界面
│   │   ├── pipeline.py             #   应用层流水线
│   │   ├── gradio_ui.py            #   Gradio UI 布局
│   │   ├── session.py              #   会话状态管理
│   │   └── data_types.py           #   应用层数据类型
│   └── utils/                      # 工具
│       ├── config.py               #   配置加载
│       └── io_utils.py             #   文件读写
├── scripts/                        # 命令行脚本
├── tests/                          # 单元测试
├── data/templates/                 # 标准动作模板库
├── models/                         # 模型文件
└── outputs/                        # 分析结果输出
```

### 数据流

```
视频/摄像头 ──→ PoseEstimator ──→ PoseSequence
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
            Preprocessing      ActionPredictor    TemplateLibrary
            (插值/平滑/归一化)   (动作分类)         (标准模板)
                    │                 │                  │
                    └─────────┬───────┘                  │
                              ▼                          ▼
                      ActionComparator ◄─── template_sequence
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              DTW对比    偏差分析    角度对比
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                      CorrectionRuleEngine
                              │
                              ▼
                      CorrectionReport
                      (评分+建议+图表)
```

---

## 8. 常见问题

### Q1: 报错 "模型文件不存在"
**A**: 确保 `models/` 目录下有 MediaPipe 模型文件（`pose_landmarker_full.task`）。可从 [Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) 下载。

### Q2: 实时模式帧率低
**A**: 
- 降低模型复杂度：修改 `configs/default.yaml` 中 `model_complexity: 0`（Lite 模式）
- 调整 `src/app/gradio_ui.py` 中的 `_PROCESS_WIDTH` 值
- 确保有 CUDA 加速

### Q3: 中文显示为方框
**A**: 系统会自动检测中文字体（SimHei、Microsoft YaHei）。如果仍显示方框，需安装中文字体。

### Q4: "模板库为空" 错误
**A**: 先运行 `python scripts/prepare_demo.py` 生成演示模板，或通过 Web 界面的"模板管理"Tab 录入标准动作。

### Q5: 摄像头无法打开
**A**: 检查 `configs/default.yaml` 中的 `camera.device_id`，Windows 通常为 0，macOS 可能为 1。

### Q6: 上传 AVI 视频显示"无法播放"/"格式不支持"
**A**: AVI 是容器格式，浏览器（Chrome/Edge）原生不支持 AVI 播放，因此 Gradio 界面无法预览。但后端使用 OpenCV + FFmpeg 处理，**不影响实际分析**。您可以直接点击"开始分析"或"上传并录入"。
- 模板录入（Tab 3）已改用文件上传方式，AVI 可正常录入。
- 如后端也报错，说明 AVI 内部编码（如特殊 MJPEG 变体）不被 OpenCV 支持，请用格式工厂或 HandBrake 转为 MP4 (H.264)。

---

> 📧 联系方式: U202215369 | 华中科技大学 人工智能与自动化学院
