## ADDED Requirements

### Requirement: Algorithm selection dropdown
系统 SHALL 在视频分析 Tab 和实时模式 Tab 中提供 DTW 算法选择下拉框，包含四个选项：「经典 DTW」「FastDTW」「DerivativeDTW」「自动选择」。

#### Scenario: User selects classic DTW
- **WHEN** 用户在算法下拉框中选择「经典 DTW」
- **THEN** 系统 SHALL 在所有分析流程（离线和实时）中使用经典 DTW 算法（algorithm="dtw"）

#### Scenario: User selects FastDTW
- **WHEN** 用户在算法下拉框中选择「FastDTW」
- **THEN** 系统 SHALL 在所有分析流程中使用 FastDTW 算法（algorithm="fastdtw"）

#### Scenario: User selects DerivativeDTW
- **WHEN** 用户在算法下拉框中选择「DerivativeDTW」
- **THEN** 系统 SHALL 在所有分析流程中使用 DerivativeDTW 算法（algorithm="ddtw"）

#### Scenario: User selects auto mode
- **WHEN** 用户在算法下拉框中选择「自动选择」
- **THEN** 系统 SHALL 在实时窗口分析中使用 FastDTW，在离线完整序列分析中使用 DerivativeDTW

### Requirement: Algorithm displayed in report
分析报告 SHALL 显示所使用的 DTW 算法名称，方便用户对比不同算法的分析结果。

#### Scenario: Report shows algorithm info
- **WHEN** 分析完成后显示报告
- **THEN** 报告 SHALL 包含「使用算法: <算法名称>」字段
