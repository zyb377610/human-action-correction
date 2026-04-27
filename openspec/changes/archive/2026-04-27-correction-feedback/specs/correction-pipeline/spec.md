## ADDED Requirements

### Requirement: 端到端自动分析
系统 SHALL 提供 CorrectionPipeline.analyze() 方法，输入 PoseSequence 输出 CorrectionReport，自动完成分类→对比→偏差分析→反馈生成全流程。

#### Scenario: 自动模式分析
- **WHEN** 用户调用 pipeline.analyze(pose_sequence)
- **THEN** 系统自动识别动作类别、匹配模板、DTW 对比、偏差分析、生成矫正报告

### Requirement: 指定模式分析
系统 SHALL 支持手动指定动作类别的降级模式，跳过分类步骤直接对比。

#### Scenario: 手动指定动作类别
- **WHEN** 用户调用 pipeline.analyze(pose_sequence, action_name="squat")
- **THEN** 系统跳过分类，直接使用 squat 类别的模板进行对比分析

### Requirement: 无模板时错误处理
系统 SHALL 在模板库为空或指定类别无模板时给出明确错误提示。

#### Scenario: 模板库无数据
- **WHEN** 指定类别的模板库为空
- **THEN** 抛出 ValueError 并提示"动作 'xxx' 无可用标准模板，请先录入模板数据"

### Requirement: 流水线可配置
系统 SHALL 支持通过配置文件或参数控制流水线行为（DTW 算法、距离度量、偏差 top-K 等）。

#### Scenario: 自定义配置
- **WHEN** 用户以 algorithm="fastdtw", top_k=3 初始化 CorrectionPipeline
- **THEN** 流水线使用 FastDTW 算法，偏差分析输出前 3 个最差关节