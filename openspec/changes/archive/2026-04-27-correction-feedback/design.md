## Context

Phase 1-4 已建成从关键点提取到动作分类的完整链路。`src/correction/` 模块目前为空，需要在其中实现矫正反馈生成的最终输出层。

**核心数据流图**：

```
PoseSequence (用户动作)
    │
    ├──→ ActionPredictor.predict()
    │        │
    │        ▼
    │    PredictionResult (label, confidence, quality_score)
    │        │
    │        ▼ 自动选择对应类别
    │    TemplateLibrary.load_all_templates(label)
    │        │
    │        ▼
    ├──→ ActionComparator.compare_with_templates()
    │        │
    │        ▼
    │    ComparisonResult (distance, similarity, path)
    │        │
    │        ▼
    ├──→ JointDeviationAnalyzer.analyze()
    │        │
    │        ▼
    │    DeviationReport (joint_deviations, worst_joints, severity)
    │        │
    │        ▼
    ├──→ AngleCalculator.compute_angles()
    │        │ 计算用户 vs 模板的关节角度差
    │        ▼
    │    AngleDeviations {joint_name: (user_angle, template_angle, diff)}
    │        │
    │        ▼
    ├──→ CorrectionRuleEngine.match_rules()
    │        │ 根据偏差类型 + 动作类别匹配规则
    │        ▼
    │    List[CorrectionItem] (关节名, 偏差描述, 矫正建议, 优先级)
    │        │
    │        ▼
    └──→ FeedbackGenerator.generate()
              │
              ▼
         CorrectionReport
         ├── 动作识别结果 (类别 + 置信度)
         ├── 质量评分 (0-100)
         ├── 整体评语 (优秀/良好/需改进)
         ├── 矫正建议列表 (按优先级排序)
         │    ├── "左膝弯曲角度不够，建议再下蹲约12°"
         │    ├── "双肩不够平稳，注意保持躯干正直"
         │    └── ...
         └── 详细偏差数据 (供可视化使用)
```

## Goals / Non-Goals

**Goals:**
- 基于规则引擎将偏差数据映射为中文自然语言矫正建议
- 精确计算关节角度差（膝角、肘角、髋角等），建议中包含具体角度数值
- 矫正建议按严重程度排序，用户优先看到最需要改进的部分
- 提供端到端 CorrectionPipeline，一键从 PoseSequence 生成完整报告
- 报告可视化：带偏差标注的骨骼图 + 结构化文本输出
- 支持无分类模型的降级模式（手动指定动作类别）

**Non-Goals:**
- 不实现基于 LLM 的自然语言生成（使用规则模板即可，不需要 GPT 接口）
- 不实现实时视频流矫正（本阶段处理单次录制的序列）
- 不实现运动处方推荐（矫正建议只针对当前动作，不涉及训练计划）

## Decisions

### D1: 矫正建议生成方式 — 规则模板 vs LLM 生成

**决定**：规则模板方案。
- 为每个动作类别定义规则集（如深蹲：膝角 < 90° → "膝盖弯曲不够"）
- 规则包含：触发条件、建议文本模板、优先级
- 文本模板支持变量插值（如"建议调整约{angle_diff:.0f}°"）

**理由**：本科毕设不需要 LLM 的灵活性；规则方案可解释、可控、无需额外 API 费用。对于 5 类运动的矫正场景，50-100 条规则完全够用。

### D2: 角度计算策略

**决定**：在 angle_utils.py 中实现基于三点向量夹角的通用角度计算。
- 定义关键角度映射表：膝角（hip-knee-ankle）、肘角（shoulder-elbow-wrist）、髋角（shoulder-hip-knee）等
- 分别计算用户和模板在对齐帧上的角度，取平均差值
- 角度差作为规则引擎的量化输入

**理由**：角度是最直观的运动学指标，用户容易理解"膝盖多弯10度"这样的建议。

### D3: 流水线降级策略

**决定**：CorrectionPipeline 支持两种模式：
- **自动模式**：使用 ActionPredictor 识别动作类别 → 自动匹配模板
- **指定模式**：用户手动指定动作类别（跳过分类，直接对比）

**理由**：分类模型可能未训练或准确率不足时，需要降级方案保证系统可用。

### D4: 报告结构设计

**决定**：CorrectionReport 包含三层信息：
1. **概要层**：动作类别 + 质量评分 + 整体评语（一句话）
2. **建议层**：按优先级排序的矫正建议列表（每条含关节名 + 偏差描述 + 具体建议）
3. **详情层**：完整的偏差数据 + 角度数据（供可视化和进一步分析）

## Risks / Trade-offs

- **[规则覆盖不全]** → 新动作类别需要手动添加规则。**缓解**：提供通用规则（任何关节偏差过大都有兜底建议），并设计可扩展的规则注册机制
- **[角度计算不精确]** → 2D 投影角度和真实 3D 角度存在差异。**缓解**：使用 MediaPipe 的 z 坐标进行 3D 角度计算，并在建议中使用"约"字标识不精确
- **[无模板数据]** → 模板库为空时流水线无法运行。**缓解**：提供明确的错误提示和使用指引