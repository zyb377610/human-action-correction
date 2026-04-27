# classification-models Specification

## Purpose
TBD - created by archiving change deep-learning-model. Update Purpose after archive.
## Requirements
### Requirement: ST-GCN 骨骼图卷积模型
系统 SHALL 实现基于人体骨骼拓扑图的时空图卷积网络（ST-GCN），接受 (batch, T, 33, 4) 输入，输出分类 logits 和质量评分。

#### Scenario: ST-GCN 前向推理
- **WHEN** 输入张量 shape 为 (B, T, 33, 4)
- **THEN** 模型输出 cls_logits shape 为 (B, num_classes)，quality_score shape 为 (B, 1)，值域 [0, 100]

#### Scenario: 骨骼邻接矩阵定义
- **WHEN** 初始化 ST-GCN 模型
- **THEN** 系统基于 MediaPipe 33 关键点定义骨骼邻接矩阵，包含自连接

### Requirement: BiLSTM-Attention 时序分类模型
系统 SHALL 实现双层双向 LSTM + Bahdanau 注意力机制的时序分类模型，接受 (batch, T, D) 输入，输出分类 logits 和质量评分。

#### Scenario: BiLSTM-Attention 前向推理
- **WHEN** 输入张量 shape 为 (B, T, D)，D 为特征维度
- **THEN** 模型输出 cls_logits shape 为 (B, num_classes)，quality_score shape 为 (B, 1)

#### Scenario: 注意力权重提取
- **WHEN** 用户调用 model.get_attention_weights()
- **THEN** 返回最近一次前向传播的注意力权重 shape (B, T)，值之和为 1

### Requirement: Transformer 序列分类模型
系统 SHALL 实现基于位置编码和多头自注意力的 Transformer Encoder 分类模型，接受 (batch, T, D) 输入，输出分类 logits 和质量评分。

#### Scenario: Transformer 前向推理
- **WHEN** 输入张量 shape 为 (B, T, D)
- **THEN** 模型输出 cls_logits shape 为 (B, num_classes)，quality_score shape 为 (B, 1)

#### Scenario: 位置编码生效
- **WHEN** 输入两个相同内容但帧顺序不同的序列
- **THEN** 模型输出不同的分类结果（位置信息影响输出）

### Requirement: 统一模型接口
系统 SHALL 定义 BaseActionModel 抽象基类，所有模型 MUST 继承该基类并实现 forward() 方法，保证输出格式一致。

#### Scenario: 模型工厂函数创建模型
- **WHEN** 用户调用 create_model(model_type="stgcn", num_classes=5) 
- **THEN** 返回对应模型实例，且该实例是 BaseActionModel 的子类

#### Scenario: 不支持的模型类型
- **WHEN** 用户调用 create_model(model_type="unknown")
- **THEN** 抛出 ValueError 并提示支持的模型类型列表

### Requirement: 双头输出架构
所有模型 SHALL 包含分类头（cls_head）和质量评分头（quality_head），共享 backbone 特征。

#### Scenario: 双头输出结构
- **WHEN** 模型完成前向推理
- **THEN** 返回包含 "cls_logits" 和 "quality_score" 两个键的字典

