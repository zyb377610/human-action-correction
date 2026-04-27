# model-training Specification

## Purpose
TBD - created by archiving change deep-learning-model. Update Purpose after archive.
## Requirements
### Requirement: 训练循环执行
系统 SHALL 提供 Trainer 类，执行完整的训练循环：前向传播 → 损失计算 → 反向传播 → 参数更新，支持多 epoch 迭代。

#### Scenario: 完整训练流程
- **WHEN** 用户调用 trainer.train(num_epochs=100)
- **THEN** 系统执行训练循环，每个 epoch 遍历训练集所有 batch，更新模型参数

### Requirement: 验证评估
系统 SHALL 在每个 epoch 结束后对验证集进行评估，记录 accuracy、loss 等指标。

#### Scenario: 每轮验证
- **WHEN** 一个 epoch 训练完成
- **THEN** 系统在验证集上计算 accuracy 和 loss，并记录到训练历史

### Requirement: 早停策略
系统 SHALL 实现 EarlyStopping 机制，当验证集指标连续 patience 个 epoch 不提升时自动停止训练。

#### Scenario: 触发早停
- **WHEN** 验证集 accuracy 连续 patience=10 个 epoch 未超过最佳值
- **THEN** 系统停止训练，并加载最佳模型权重

#### Scenario: 正常收敛不触发早停
- **WHEN** 验证集 accuracy 在 patience 窗口内有提升
- **THEN** 系统继续训练，更新最佳模型检查点

### Requirement: 学习率调度
系统 SHALL 支持 CosineAnnealingLR 学习率调度器，在训练过程中动态调整学习率。

#### Scenario: 余弦退火调度
- **WHEN** 训练进行到第 t 个 epoch
- **THEN** 学习率按余弦退火公式衰减

### Requirement: 模型检查点管理
系统 SHALL 在训练过程中保存最佳模型和最新模型的 checkpoint，包含模型权重、优化器状态和训练指标。

#### Scenario: 保存最佳检查点
- **WHEN** 验证集 accuracy 创新高
- **THEN** 系统保存 best_model.pth 到 checkpoints 目录，包含 model_state_dict、optimizer_state_dict、epoch、best_accuracy

#### Scenario: 加载检查点恢复训练
- **WHEN** 用户调用 trainer.load_checkpoint(path)
- **THEN** 系统恢复模型权重、优化器状态和训练进度，可继续训练

### Requirement: 多任务损失函数
系统 SHALL 实现组合损失函数：L_total = L_cls + λ × L_quality，其中 L_cls 为 CrossEntropyLoss，L_quality 为 MSELoss。

#### Scenario: 损失计算
- **WHEN** 模型输出分类 logits 和质量评分
- **THEN** 系统分别计算分类损失和质量评分损失，按 lambda_quality 权重加权求和

### Requirement: 训练日志与曲线
系统 SHALL 记录训练过程的 loss、accuracy 等指标，并支持导出训练曲线图。

#### Scenario: 绘制训练曲线
- **WHEN** 训练完成后调用 trainer.plot_training_curves()
- **THEN** 生成包含 train_loss、val_loss、train_acc、val_acc 的折线图并保存

### Requirement: 可复现训练
系统 SHALL 支持通过设置随机种子保证训练可复现。

#### Scenario: 固定随机种子
- **WHEN** 用户设置 seed=42 后运行两次训练
- **THEN** 两次训练的 loss 和 accuracy 曲线完全一致

