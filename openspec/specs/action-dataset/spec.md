# action-dataset Specification

## Purpose
TBD - created by archiving change deep-learning-model. Update Purpose after archive.
## Requirements
### Requirement: 数据集从模板库构建
系统 SHALL 从 TemplateLibrary 加载所有动作类别的模板数据，结合 DataAugmentor 生成增强样本，构建 PyTorch Dataset 对象。

#### Scenario: 加载模板库生成数据集
- **WHEN** 用户以模板目录路径和增强配置初始化 ActionDataset
- **THEN** 系统加载所有类别的模板，每个模板生成 augment_per_template 个增强样本，总样本数 = 模板数 × (1 + augment_per_template)

### Requirement: 序列长度标准化
系统 SHALL 将所有关键点序列统一为固定长度 target_frames，短序列用零填充（pad），长序列截断（truncate）。

#### Scenario: 短序列填充
- **WHEN** 输入序列长度 T < target_frames
- **THEN** 系统在序列末尾填充零帧至 target_frames 长度，并生成对应的 padding mask

#### Scenario: 长序列截断
- **WHEN** 输入序列长度 T > target_frames
- **THEN** 系统截断序列至 target_frames 长度（保留前 target_frames 帧）

### Requirement: 训练/验证/测试划分
系统 SHALL 支持按指定比例（默认 70/15/15）将数据集划分为训练集、验证集和测试集，并支持固定随机种子保证可复现。

#### Scenario: 按比例划分数据集
- **WHEN** 用户指定 split_ratios=(0.7, 0.15, 0.15) 和 random_seed=42
- **THEN** 系统返回三个不重叠的子数据集，且每次运行结果一致

### Requirement: 输入模式切换
系统 SHALL 支持两种输入模式：原始关键点模式（输出 shape: (T, 33, 4)）和特征向量模式（输出 shape: (T, D)，通过 FeatureExtractor 计算）。

#### Scenario: 原始关键点模式
- **WHEN** ActionDataset 的 input_mode 设为 "keypoints"
- **THEN** __getitem__ 返回张量 shape 为 (target_frames, 33, 4)

#### Scenario: 特征向量模式
- **WHEN** ActionDataset 的 input_mode 设为 "features"
- **THEN** __getitem__ 返回张量 shape 为 (target_frames, D)，D 为特征维度

### Requirement: DataLoader 封装
系统 SHALL 提供 create_dataloaders() 工厂函数，返回训练/验证/测试三个 DataLoader，支持 batch_size、shuffle、num_workers 配置。

#### Scenario: 创建 DataLoader
- **WHEN** 用户调用 create_dataloaders(templates_dir, batch_size=32)
- **THEN** 返回 (train_loader, val_loader, test_loader) 三元组，训练集 shuffle=True，验证/测试集 shuffle=False

