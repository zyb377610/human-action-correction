# correction-report Specification

## Purpose
TBD - created by archiving change correction-feedback. Update Purpose after archive.
## Requirements
### Requirement: CorrectionReport 结构
系统 SHALL 生成包含三层信息的 CorrectionReport：概要层（类别+评分+评语）、建议层（排序的矫正建议列表）、详情层（偏差+角度数据）。

#### Scenario: 生成完整报告
- **WHEN** FeedbackGenerator 接收分类结果、DTW 对比结果和偏差分析报告
- **THEN** 输出 CorrectionReport 包含所有三层信息

### Requirement: 整体评语生成
系统 SHALL 根据质量评分自动生成整体评语：≥90 优秀、70-89 良好、50-69 需改进、<50 较差。

#### Scenario: 优秀评语
- **WHEN** 质量评分为 92
- **THEN** 整体评语为"动作完成质量优秀，继续保持！"

#### Scenario: 需改进评语
- **WHEN** 质量评分为 58
- **THEN** 整体评语包含改进建议

### Requirement: 矫正建议优先级排序
系统 SHALL 按优先级（高/中/低）对矫正建议排序，同等优先级内按偏差程度降序。

#### Scenario: 按优先级排序
- **WHEN** 存在高优先级建议1条和中优先级建议2条
- **THEN** 报告中高优先级建议排在最前

### Requirement: 报告文本输出
系统 SHALL 提供 to_text() 方法，将 CorrectionReport 格式化为可读的中文文本。

#### Scenario: 文本格式输出
- **WHEN** 调用 report.to_text()
- **THEN** 返回包含动作名称、评分、评语和所有建议的格式化文本

### Requirement: 报告 JSON 序列化
系统 SHALL 提供 to_dict() 方法，将 CorrectionReport 序列化为字典，便于 API 输出。

#### Scenario: JSON 序列化
- **WHEN** 调用 report.to_dict()
- **THEN** 返回可 json.dumps() 的字典

