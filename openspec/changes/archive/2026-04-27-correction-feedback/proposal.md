## Why

Phase 1-4 已完成：姿态估计（关键点提取 + 特征工程）→ 标准动作数据层（模板库 + 预处理 + 增强）→ DTW 对比（经典/FastDTW/DDTW + 逐关节偏差分析）→ 深度学习分类模型（ST-GCN/BiLSTM/Transformer + 质量评分）。

当前系统能够：识别动作类别、计算相似度、定位偏差最大的关节。但**缺少最后一环**——将这些底层分析结果转化为**用户可理解的自然语言矫正建议**。用户看到的不应是 "left_knee 偏差 0.15"，而是"你的左膝弯曲角度不够，建议再下蹲约10度"。

Phase 5（矫正反馈生成）是系统的**最终输出层**，将所有模块串联成端到端流水线：视频输入 → 姿态估计 → 自动分类 → 模板匹配 → DTW 对比 → 偏差分析 → **生成矫正反馈报告**。

**预计开发时间：10-14 小时**

## What Changes

- **新增矫正规则引擎** `src/correction/rules.py`：基于关节偏差的规则匹配系统，将偏差数据映射为具体矫正动作描述
- **新增反馈报告生成器** `src/correction/feedback.py`：将规则引擎输出组织为结构化 CorrectionReport（含总评 + 逐关节建议 + 优先级排序）
- **新增端到端分析流水线** `src/correction/pipeline.py`：串联所有模块的 CorrectionPipeline，输入 PoseSequence → 输出 CorrectionReport
- **新增角度计算工具** `src/correction/angle_utils.py`：计算特定关节角度（膝关节角、肘关节角等），用于生成精确的角度偏差建议
- **新增矫正反馈数据类型** `src/correction/data_types.py`：CorrectionReport、CorrectionItem、JointAdvice 等数据结构
- **新增反馈可视化** `src/correction/report_visualizer.py`：将矫正报告渲染为带标注的骨骼图 + 文本报告
- **新增端到端演示脚本** `scripts/run_correction.py`：从视频文件到矫正报告的完整流程脚本

## Capabilities

### New Capabilities
- `correction-rules`: 矫正规则引擎——基于关节偏差和角度差异的规则匹配，输出结构化矫正建议
- `correction-report`: 反馈报告生成——整合分类结果 + DTW 对比 + 偏差分析，输出用户友好的 CorrectionReport
- `correction-pipeline`: 端到端矫正流水线——从 PoseSequence 到 CorrectionReport 的一键分析
- `correction-visualization`: 矫正反馈可视化——骨骼标注图 + 文本报告输出

### Modified Capabilities
<!-- 无已有 capability 需要修改 -->

## Impact

- **新增代码**：`src/correction/` 目录下新增 ~6 个 Python 模块
- **新增脚本**：`scripts/run_correction.py`
- **新增测试**：`tests/test_correction.py`
- **上游依赖**：`src/pose_estimation/`（PoseSequence、FeatureExtractor）、`src/action_comparison/`（ActionComparator、JointDeviationAnalyzer、DeviationReport）、`src/models/`（ActionPredictor、PredictionResult）、`src/data/`（TemplateLibrary）
- **下游影响**：为后续 UI/Web 应用层提供完整的矫正 API
- **`src/correction/__init__.py`** 需更新导出