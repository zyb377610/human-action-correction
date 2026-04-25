# 基于深度学习的人体动作矫正系统

> **毕业设计** | CS2201 | 张玉倍 | U202215369

## 📌 项目简介

本项目旨在设计并实现一个基于深度学习的人体动作矫正系统。系统通过摄像头捕获用户的运动视频，利用人体姿态估计技术提取骨骼关键点，结合动态时间规整（DTW）等算法与标准动作进行对比分析，实时检测动作偏差并给出矫正建议。

### 核心技术栈

| 技术方向 | 关键技术 | 说明 |
|---------|---------|------|
| 姿态估计 | MediaPipe / OpenPose | 从视频中提取人体骨骼关键点 |
| 动作对比 | DTW (Dynamic Time Warping) | 时间序列相似度度量，支持不等长序列对比 |
| 深度学习 | CNN / LSTM / Transformer | 动作特征提取与分类 |
| 前端展示 | Web / Desktop GUI | 实时可视化与交互反馈 |

## 🏗️ 项目结构

```
BysjOnGitHub/
├── docs/                       # 文档资料
│   ├── proposal/               # 开题报告、任务书、答辩PPT
│   ├── references/             # 参考文献
│   └── design/                 # 系统设计文档
├── src/                        # 源代码
│   ├── pose_estimation/        # 姿态估计模块
│   ├── action_comparison/      # 动作对比模块（DTW等）
│   ├── correction/             # 动作矫正与反馈模块
│   ├── models/                 # 深度学习模型定义
│   ├── utils/                  # 工具函数
│   └── app/                    # 应用入口与UI
├── configs/                    # 配置文件
├── data/                       # 数据目录
│   ├── samples/                # 示例数据（可追踪）
│   ├── raw/                    # 原始数据（gitignore）
│   └── processed/              # 处理后数据（gitignore）
├── notebooks/                  # Jupyter 实验笔记本
├── tests/                      # 单元测试
├── scripts/                    # 辅助脚本
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
└── .gitignore                  # Git 忽略规则
```

## 🚀 快速开始

### 环境要求

- Python >= 3.9
- CUDA >= 11.8（推荐，用于 GPU 加速）
- Webcam 或视频文件

### 安装

```bash
# 克隆仓库
git clone https://github.com/<your-username>/human-action-correction.git
cd human-action-correction

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# TODO: 补充运行指令
python src/app/main.py
```

## 📚 参考文献

1. **MediaPipe: A Framework for Building Perception Pipelines**
   - Lugaresi C, Tang J, Nash H, et al. Google Research, 2019.
   - MediaPipe 框架用于构建感知管道，支持跨平台部署，是本项目姿态估计的核心工具之一。

2. **Dynamic Time Warping (DTW) Algorithm in Speech: A Review**
   - Yadav M, Alam M A. IJRECE, 2018.
   - DTW 算法综述，涵盖经典 DTW 及其多种变体（Sparse DTW、Fast DTW、Derivative DTW 等），为动作序列对比提供理论基础。

3. **Early Diagnosis of Autism: A Review of Video-Based Motion Analysis and Deep Learning Techniques**
   - Yang Z, Zhang Y, Ning J, et al. IEEE Access, 2025.
   - 基于视频的运动分析与深度学习综述，涵盖姿态估计、特征提取、分类模型等，为系统整体架构设计提供参考。

## 📋 开发计划

- [ ] 项目初始化与环境搭建
- [ ] 姿态估计模块开发（基于 MediaPipe）
- [ ] 标准动作数据采集与预处理
- [ ] DTW 动作对比算法实现
- [ ] 深度学习动作分类模型训练
- [ ] 矫正反馈模块开发
- [ ] 前端界面与实时展示
- [ ] 系统集成测试
- [ ] 毕业论文撰写

## 📄 License

本项目仅用于学术研究和毕业设计，暂不开放商业使用。
