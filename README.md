# 基于深度学习的人体动作矫正系统

> **毕业设计** | CS2201 | 张玉倍 | U202215369

## 📌 项目简介

本项目实现了一个基于深度学习的人体动作矫正系统。系统通过摄像头或视频输入，利用 MediaPipe 姿态估计技术提取 33 个人体骨骼关键点，结合 DTW 动态时间规整算法与标准动作模板进行对比分析，通过规则引擎生成中文自然语言矫正建议，并提供基于 Gradio 的 Web 交互界面。

### 核心技术栈

| 技术方向 | 关键技术 | 说明 |
|---------|---------|------|
| 姿态估计 | MediaPipe PoseLandmarker | 从视频中提取 33 个人体骨骼关键点 |
| 动作对比 | DTW (Dynamic Time Warping) | 时间序列相似度度量，支持不等长序列对比 |
| 深度学习 | ST-GCN / BiLSTM-Attention / Transformer | 动作分类与特征提取 |
| 矫正反馈 | 规则引擎 + 模板匹配 | 50+ 条专家规则，中文自然语言矫正建议 |
| Web 界面 | Gradio | 视频上传分析、实时摄像头、模板管理 |

## 🏗️ 项目结构

```
human-action-correction/
├── configs/                    # 配置文件
│   └── default_config.yaml     # 系统默认配置
├── docs/                       # 文档资料
│   ├── proposal/               # 开题报告、任务书、答辩PPT
│   ├── references/             # 参考文献
│   └── design/                 # 系统设计文档
├── src/                        # 源代码
│   ├── pose_estimation/        # Phase 1: 姿态估计模块 (MediaPipe)
│   ├── data/                   # Phase 2: 数据预处理与模板库
│   ├── action_comparison/      # Phase 3: DTW 动作对比模块
│   ├── models/                 # Phase 4: 深度学习分类模型
│   ├── correction/             # Phase 5: 矫正反馈模块
│   ├── app/                    # Phase 6: Web 应用界面
│   └── utils/                  # 工具函数与配置管理
├── scripts/                    # 辅助脚本
│   ├── launch_app.py           # 一键启动 Web 界面
│   ├── prepare_demo.py         # 演示数据准备
│   ├── batch_process.py        # 离线批处理
│   ├── train_model.py          # 模型训练
│   └── evaluate_model.py       # 模型评估
├── tests/                      # 单元测试 & 集成测试
├── data/                       # 数据目录
│   ├── templates/              # 标准动作模板库
│   └── samples/                # 示例数据
├── models/                     # 模型文件
├── requirements.txt            # Python 依赖
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python >= 3.9
- CUDA >= 11.8（推荐，用于 GPU 加速）
- Webcam 或视频文件

### 安装

```bash
# 克隆仓库
git clone https://github.com/zyb377610/human-action-correction.git
cd human-action-correction

# 安装依赖
pip install -r requirements.txt
```

### 准备演示数据

```bash
# 生成 5 种标准动作的合成模板
python scripts/prepare_demo.py
```

### 启动 Web 界面

```bash
# 默认启动 (端口 7860)
python scripts/launch_app.py

# 指定模板目录和端口
python scripts/launch_app.py --templates data/templates --port 8080

# 生成公网分享链接
python scripts/launch_app.py --share
```

启动后在浏览器访问 `http://localhost:7860`，可使用四个功能 Tab：
- **📹 视频分析** — 上传视频获取矫正报告
- **📷 实时模式** — 摄像头实时骨骼叠加 + 录制分析
- **📋 模板管理** — 查看/录入标准动作模板
- **ℹ️ 系统说明** — 使用指南和技术架构

### 离线批处理

```bash
python scripts/batch_process.py --input videos/ --output results/
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## 📋 开发计划

- [x] 项目初始化与环境搭建
- [x] Phase 1: 姿态估计模块开发（MediaPipe 33 关键点）
- [x] Phase 2: 标准动作数据采集与预处理
- [x] Phase 3: DTW 动作对比算法实现
- [x] Phase 4: 深度学习动作分类模型（ST-GCN / BiLSTM / Transformer）
- [x] Phase 5: 矫正反馈模块（规则引擎 + 自然语言报告）
- [x] Phase 6: Web 界面与实时展示（Gradio）
- [x] Phase 7: 系统集成测试与部署准备
- [ ] 毕业论文撰写

## 📚 参考文献

1. **MediaPipe: A Framework for Building Perception Pipelines**
   - Lugaresi C, Tang J, Nash H, et al. Google Research, 2019.

2. **Dynamic Time Warping (DTW) Algorithm in Speech: A Review**
   - Yadav M, Alam M A. IJRECE, 2018.

3. **Early Diagnosis of Autism: A Review of Video-Based Motion Analysis and Deep Learning Techniques**
   - Yang Z, Zhang Y, Ning J, et al. IEEE Access, 2025.

## 📄 License

本项目仅用于学术研究和毕业设计，暂不开放商业使用。