## MODIFIED Requirements

### Requirement: 端到端自动分析
系统 SHALL 提供 CorrectionPipeline.analyze() 方法，输入 PoseSequence 输出 CorrectionReport，自动完成分类→对比→偏差分析→反馈生成全流程。支持可选的进度回调参数。

#### Scenario: 自动模式分析
- **WHEN** 用户调用 pipeline.analyze(pose_sequence)
- **THEN** 系统自动识别动作类别、匹配模板、DTW 对比、偏差分析、生成矫正报告

#### Scenario: 进度回调
- **WHEN** 用户调用 pipeline.analyze(pose_sequence, progress_callback=fn)
- **THEN** 系统在每个处理阶段调用 fn(step, total, message)，报告当前进度
