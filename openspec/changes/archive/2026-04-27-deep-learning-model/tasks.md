## 1. 基础设施与数据类型

- [x] 1.1 实现 `src/models/data_types.py` — 定义 PredictionResult dataclass（label, confidence, quality_score, class_probs）和 ModelConfig dataclass。验证：PredictionResult 可正常实例化并访问所有属性
- [x] 1.2 扩展 `configs/default.yaml` — 扩展 model 配置段，新增 backbone_type（stgcn/bilstm/transformer）、hidden_dim、num_layers、dropout、lambda_quality 等超参。验证：load_config() 能正确读取所有新增配置项
- [x] 1.3 实现 `src/models/skeleton_graph.py` — 定义 MediaPipe 33 关键点的骨骼邻接矩阵（含自连接），提供 get_adjacency_matrix() 和 get_edge_list()。验证：邻接矩阵为 33×33 对称矩阵，对角线全为 1

## 2. 数据集构建

- [x] 2.1 实现 `src/models/dataset.py` — ActionDataset 类，从 TemplateLibrary 加载模板数据 + DataAugmentor 增强，返回 (序列张量, 类别标签, 质量评分)。验证：len(dataset) == 模板数 × (1 + augment_per_template)
- [x] 2.2 实现序列 pad/truncate — ActionDataset 将所有序列统一到 target_frames 长度，生成 padding mask。验证：输出张量 shape 为 (target_frames, 33, 4) 或 (target_frames, D)
- [x] 2.3 实现输入模式切换 — 支持 "keypoints" 和 "features" 两种 input_mode。验证：keypoints 模式输出 (T, 33, 4)，features 模式输出 (T, D)
- [x] 2.4 实现 train/val/test 划分和 create_dataloaders() 工厂函数。验证：三个 DataLoader 的样本数之和等于总数据量，训练集 shuffle=True

## 3. 模型架构实现

- [x] 3.1 实现 `src/models/base_model.py` — BaseActionModel 抽象基类，定义 forward() 接口（返回 dict 含 cls_logits 和 quality_score）。验证：抽象类不可直接实例化
- [x] 3.2 实现 `src/models/stgcn.py` — ST-GCN 模型：3 层时空图卷积 + 全局平均池化 + 双头输出。验证：输入 (B, T, 33, 4) 输出 cls_logits (B, num_classes) 和 quality_score (B, 1)
- [x] 3.3 实现 `src/models/bilstm.py` — BiLSTM-Attention 模型：双层 BiLSTM + Bahdanau 注意力 + 双头输出，支持 get_attention_weights()。验证：输入 (B, T, D) 输出正确 shape，注意力权重和为 1
- [x] 3.4 实现 `src/models/transformer_model.py` — Transformer Encoder 分类模型：位置编码 + 2 层 4 头注意力 + 双头输出。验证：输入 (B, T, D) 输出正确 shape，不同帧顺序产生不同输出
- [x] 3.5 实现 `src/models/model_factory.py` — create_model() 工厂函数，根据 model_type 创建对应模型实例。验证：create_model(&quot;stgcn&quot;) 返回 STGCN 实例，create_model(&quot;unknown&quot;) 抛出 ValueError

## 4. 训练流水线

- [x] 4.1 实现 `src/models/trainer.py` — Trainer 类核心训练循环：train_epoch() + validate_epoch()，支持 AdamW 优化器。验证：一个 epoch 训练后 loss 有变化
- [x] 4.2 实现多任务损失函数 — MultiTaskLoss 类：L_total = L_cls + λ × L_quality。验证：给定固定输入，损失值与手动计算一致
- [x] 4.3 实现早停策略 — EarlyStopping 类，监控 val_accuracy，patience=10。验证：连续 10 次不提升时 early_stop 标志为 True
- [x] 4.4 实现学习率调度和 checkpoint 管理 — CosineAnnealingLR 调度 + save/load checkpoint。验证：保存后加载 checkpoint，模型参数完全一致
- [x] 4.5 实现训练日志与曲线 — 记录每 epoch 的 train_loss、val_loss、train_acc、val_acc，提供 plot_training_curves() 方法。验证：生成训练曲线图 PNG 文件
- [x] 4.6 实现可复现训练 — set_seed() 固定 Python/NumPy/PyTorch 随机种子。验证：相同种子两次训练 loss 序列一致

## 5. 推理接口

- [x] 5.1 实现 `src/models/predictor.py` — ActionPredictor 类：加载 checkpoint + predict(PoseSequence) → PredictionResult。验证：加载训练好的模型后对 PoseSequence 返回正确的 PredictionResult
- [x] 5.2 实现批量推理 — predict_batch() 方法，支持列表输入。验证：输入 N 个 PoseSequence 返回 N 个 PredictionResult
- [x] 5.3 实现设备自动检测 — CUDA/CPU 自适应。验证：在 CPU 环境下推理不报错

## 6. 模块集成与导出

- [x] 6.1 更新 `src/models/__init__.py` — 导出所有公共 API（BaseActionModel, STGCN, BiLSTMAttention, TransformerClassifier, ActionDataset, Trainer, ActionPredictor, PredictionResult, create_model）。验证：from src.models import ActionPredictor 可正常导入
- [x] 6.2 编写 `tests/test_models.py` — 单元测试覆盖数据集、三种模型前向传播、训练器单步训练、推理器预测。验证：所有测试通过

## 7. 脚本与端到端演示

- [x] 7.1 编写 `scripts/train_model.py` — 训练脚本：解析命令行参数（模型类型/epochs/batch_size 等）→ 构建数据集 → 训练 → 保存模型 + 训练曲线。验证：脚本运行后生成 checkpoint 文件和训练曲线图
- [x] 7.2 编写 `scripts/evaluate_model.py` — 评估脚本：加载模型 → 测试集评估 → 输出 accuracy、precision、recall、F1、混淆矩阵。验证：脚本运行后打印分类报告并保存混淆矩阵图