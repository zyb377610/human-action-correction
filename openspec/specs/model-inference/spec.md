# model-inference Specification

## Purpose
TBD - created by archiving change deep-learning-model. Update Purpose after archive.
## Requirements
### Requirement: 统一推理接口
系统 SHALL 提供 ActionPredictor 类，封装模型加载和推理逻辑，输入 PoseSequence 输出 PredictionResult。

#### Scenario: 单样本推理
- **WHEN** 用户调用 predictor.predict(pose_sequence)
- **THEN** 返回 PredictionResult，包含 label（类别名称）、confidence（置信度 0-1）、quality_score（质量评分 0-100）、class_probs（各类别概率字典）

#### Scenario: 批量推理
- **WHEN** 用户调用 predictor.predict_batch(pose_sequences_list)
- **THEN** 返回 PredictionResult 列表，长度等于输入列表长度

### Requirement: 模型权重加载
系统 SHALL 支持从 checkpoint 文件加载训练好的模型权重进行推理。

#### Scenario: 加载最佳模型
- **WHEN** 用户以 checkpoint 路径初始化 ActionPredictor
- **THEN** 系统加载模型权重并设置为 eval 模式，准备推理

#### Scenario: checkpoint 文件不存在
- **WHEN** 用户提供不存在的 checkpoint 路径
- **THEN** 抛出 FileNotFoundError 并给出明确提示

### Requirement: 预测结果数据结构
系统 SHALL 定义 PredictionResult dataclass，包含分类标签、置信度、质量评分和各类概率。

#### Scenario: PredictionResult 属性访问
- **WHEN** 获得 PredictionResult 对象
- **THEN** 可访问 result.label（str）、result.confidence（float）、result.quality_score（float）、result.class_probs（dict）

### Requirement: 设备自动检测
系统 SHALL 自动检测 CUDA 可用性，在 GPU 可用时使用 GPU 推理，否则使用 CPU。

#### Scenario: GPU 可用
- **WHEN** 系统检测到 CUDA 可用
- **THEN** 模型和输入数据自动迁移到 GPU

#### Scenario: 仅 CPU 可用
- **WHEN** 系统无 CUDA 设备
- **THEN** 模型和推理在 CPU 上执行，无报错

### Requirement: 推理延迟约束
系统 SHALL 保证单样本推理延迟 < 50ms（CPU 环境下），以支持实时应用场景。

#### Scenario: 实时推理性能
- **WHEN** 在 CPU 环境下对单个 PoseSequence 调用 predict()
- **THEN** 推理完成时间 < 50ms

