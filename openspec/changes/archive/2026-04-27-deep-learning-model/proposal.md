## Why

Phase 1-3 已完成姿态估计（33 关键点提取 + 特征工程）、标准动作数据层（模板库 + 预处理 + 数据增强）和 DTW 动作对比（经典/FastDTW/DDTW + 逐关节偏差分析）。DTW 方案在序列对齐方面表现良好，但存在两个局限：（1）对动作类别无法自动识别——需要人工指定与哪个模板对比；（2）对复杂非线性时空模式的建模能力有限。

深度学习分类模型（Phase 4）旨在**自动识别用户正在执行的动作类别**，并提供基于学习的**动作质量评分**。这一能力是 Phase 5（矫正反馈）的核心前置：先识别动作 → 自动匹配对应模板 → DTW 精细对比 → 生成矫正建议。参考 ASD 运动分析综述，CNN-LSTM 时空特征提取和 Transformer 自注意力架构在骨骼关键点动作识别中表现优异，适合本项目的关键点序列输入。

**预计开发时间：12-16 小时**

## What Changes

- **新增动作分类数据集工具**：基于现有 TemplateLibrary 和数据增强模块，构建训练/验证/测试数据集（PyTorch Dataset + DataLoader）
- **新增 ST-GCN 骨骼图卷积模型**：基于人体骨骼拓扑图的时空图卷积网络，直接对关键点序列建模
- **新增 LSTM 时序分类模型**：双层 BiLSTM + 注意力机制，处理变长关键点特征序列
- **新增 Transformer 分类模型**：基于位置编码 + 多头自注意力的序列分类器，捕获长距离时序依赖
- **新增模型训练流水线**：支持训练、验证、早停、学习率调度、模型保存/加载
- **新增动作质量评分头**：在分类基础上输出 0-100 的动作质量分数
- **新增模型推理接口**：统一的 predict() API，输入 PoseSequence 输出分类标签 + 置信度 + 质量评分
- **新增训练与评估脚本**：`scripts/train_model.py`、`scripts/evaluate_model.py`

## Capabilities

### New Capabilities
- `action-dataset`: 动作分类数据集构建——从模板库 + 数据增强生成 PyTorch Dataset，支持训练/验证/测试划分、序列填充与截断、数据归一化
- `classification-models`: 深度学习动作分类模型——ST-GCN / BiLSTM-Attention / Transformer 三种架构，支持配置化选择与超参调整
- `model-training`: 模型训练流水线——训练循环、验证评估、早停策略、学习率调度、checkpoints 管理、训练日志与曲线
- `model-inference`: 模型推理接口——统一 predict() API 输入 PoseSequence 输出分类标签 + 置信度 + 质量评分，支持批量推理

### Modified Capabilities
<!-- 无已有 capability 需要修改 -->

## Impact

- **新增代码**：`src/models/` 目录下新增 ~8 个 Python 模块（数据集、三种模型、训练器、推理器、工具函数）
- **新增脚本**：`scripts/train_model.py`、`scripts/evaluate_model.py`
- **新增测试**：`tests/test_models.py`
- **依赖确认**：`torch>=2.0.0`、`torchvision>=0.15.0`（已在 requirements.txt）；新增 `tensorboard` 用于训练可视化（可选）
- **配置扩展**：`configs/default.yaml` 中 `model` 配置段需扩展，支持多模型架构选择和训练超参
- **上游依赖**：`src/pose_estimation/data_types.py`（PoseSequence）、`src/data/`（TemplateLibrary、预处理、数据增强）
- **下游影响**：Phase 5（矫正反馈）将消费分类结果自动匹配对应标准模板进行 DTW 对比
- **存储影响**：模型权重文件（~10-50MB）保存在 `outputs/checkpoints/`