## Context

Phase 1-3 已建成完整的数据流水线：视频 → MediaPipe 关键点提取（33 点 × 4 维） → 特征工程（角度/骨骼比/速度） → 标准模板库 → DTW 对比。当前 `src/models/` 模块为空，需在其中实现深度学习分类模型，从关键点序列中自动识别动作类别并评估执行质量。

**数据流图**：

```
PoseSequence (T, 33, 4)
    │
    ├─── ActionDataset ──→ DataLoader (batch, T_pad, C)
    │        │                   │
    │     [pad/truncate]    [normalize]
    │        │                   │
    │        ▼                   ▼
    │   ┌─────────────────────────────┐
    │   │   Model Backbone (可选)      │
    │   │  ┌──────────┐              │
    │   │  │ ST-GCN   │  (图卷积)    │
    │   │  ├──────────┤              │
    │   │  │ BiLSTM   │  (时序建模)  │
    │   │  ├──────────┤              │
    │   │  │Transformer│ (自注意力)   │
    │   │  └──────────┘              │
    │   │       │                     │
    │   │       ▼                     │
    │   │  Feature Vector (D)         │
    │   │    ┌────┴────┐              │
    │   │    ▼         ▼              │
    │   │ cls_head   quality_head     │
    │   │  (softmax)  (sigmoid×100)   │
    │   └─────────────────────────────┘
    │        │              │
    │        ▼              ▼
    │   类别标签+置信度   质量评分(0-100)
    │
    └─── ActionPredictor (统一推理接口)
              │
              ▼
         PredictionResult(label, confidence, quality_score)
```

**输入格式**：`PoseSequence` 对象 → numpy 数组 `(T, 33, 4)`，其中 4 维 = (x, y, z, visibility)
**输出格式**：`PredictionResult(label: str, confidence: float, quality_score: float, class_probs: dict)`

## Goals / Non-Goals

**Goals:**
- 实现至少 3 种动作分类模型（ST-GCN、BiLSTM-Attention、Transformer），支持通过配置文件切换
- 构建基于现有模板库和数据增强的 PyTorch 数据集，支持训练/验证/测试划分
- 提供完整的训练流水线，包含早停、学习率调度、checkpoints 管理
- 统一的推理接口，输入 PoseSequence 输出分类+质量评分
- 训练可复现：固定随机种子，记录所有超参
- 支持 CPU 和 GPU 训练（自动检测 CUDA）

**Non-Goals:**
- 不实现在线学习/增量训练（离线训练后加载权重即可）
- 不实现模型蒸馏或量化（本科毕设阶段不需要部署优化）
- 不实现视频级别端到端模型（输入是已提取的关键点序列，非原始视频帧）
- 不实现 GAN 生成式数据增强（使用已有的数据增强模块即可）

## Decisions

### D1: 输入表示选择 — 原始关键点 vs 特征工程后的向量

**决定**：两者都支持，通过配置切换。
- **原始关键点模式**：输入 `(T, 33, 4)`，让模型自行学习空间关系（适合 ST-GCN）
- **特征向量模式**：输入 `(T, D)`，使用 feature_extractor 提取角度+比例+速度特征（适合 LSTM/Transformer）

**理由**：ST-GCN 天然适合图结构的原始坐标输入；LSTM/Transformer 更适合紧凑的特征向量。两种模式互补，便于在毕设论文中对比分析。

### D2: 模型架构选择 — ST-GCN / BiLSTM / Transformer

**决定**：三种均实现，作为毕设方法对比。
- **ST-GCN**：参考 Yan et al. 2018，基于人体骨骼拓扑图（MediaPipe 33 关键点邻接矩阵），3 层时空图卷积 + 全局平均池化
- **BiLSTM-Attention**：双层双向 LSTM + Bahdanau 注意力，参考 ASD 综述中 CNN-LSTM 时空特征方案
- **Transformer**：2 层 Encoder，4 头注意力，位置编码，参考 ViT 思路适配 1D 序列

**理由**：三种模型分别代表图神经网络、循环神经网络和自注意力三大范式，覆盖毕设论文的方法对比需求。每种模型参数量控制在 1-5M，可在 CPU 上训练。

**备选方案**：仅实现 LSTM — 但论文方法对比不够丰富，遂否决。

### D3: 质量评分机制 — 分类头 + 回归头

**决定**：双头网络设计。共享 backbone 的特征向量后分成两个输出头：
- `cls_head`：`Linear(D, num_classes)` → softmax → 分类概率
- `quality_head`：`Linear(D, 1)` → sigmoid × 100 → 质量评分 [0, 100]

**理由**：分类和质量评估共享时空特征，双头设计既减少计算量又可进行多任务学习。质量评分可作为标准/非标准二分类的回归细化。

**备选方案**：单独训练质量评估模型 — 会导致训练复杂度增加，且需要额外的质量标注数据，否决。

### D4: 数据集构建策略

**决定**：基于现有 TemplateLibrary 的模板 + DataAugmentor 的增强样本。
- 每个动作类别从模板库加载所有模板
- 每个模板通过数据增强（时间拉伸、噪声、镜像）扩充 N 倍（默认 augment_per_template=5）
- 按 70/15/15 划分训练/验证/测试
- 序列统一 pad/truncate 至 `default_target_frames`（默认 60 帧）

**理由**：利用已有的数据基础设施，无需引入外部数据集。对于毕设的 5 类动作分类任务，增强后的数据量足够训练轻量模型。

### D5: 训练策略

**决定**：
- 优化器：AdamW（lr=1e-3, weight_decay=1e-4）
- 调度器：CosineAnnealingLR（T_max=epochs）
- 早停：patience=10，监控验证集 accuracy
- 损失函数：分类用 CrossEntropyLoss，质量评分用 MSELoss，总损失 = L_cls + λ × L_quality（λ=0.5）
- 批大小：32（可在配置文件调整）
- 最大 epochs：100（通常早停在 30-50）

## Risks / Trade-offs

- **[数据量不足]** → 模板库数据有限，可能导致过拟合。**缓解**：数据增强扩充 5-10 倍 + Dropout(0.3) + 权重衰减
- **[质量评分缺乏真实标注]** → 训练数据中标准模板的质量评分默认为 100，增强样本根据变形程度设置 70-95。**缓解**：可后续收集真实用户数据微调
- **[ST-GCN 骨骼图定义]** → MediaPipe 33 关键点的邻接矩阵需手动定义。**缓解**：参考 MediaPipe 官方文档定义骨骼连接
- **[GPU 不可用]** → 部分用户环境无 CUDA。**缓解**：所有模型参数量 < 5M，CPU 训练可行（预计 10-30 分钟/100 epochs）
- **[模型推理延迟]** → 实时应用需要 < 50ms 推理。**缓解**：轻量模型设计 + batch_size=1 推理优化