## MODIFIED Requirements

### Requirement: Realtime mode tab interaction flow
实时模式 Tab SHALL 提供以下交互流程：选择动作类型 → 点击开始 → 3 秒倒计时 → 实时跟做（边做边显示即时矫正建议）→ 点击结束 → 显示总体评分和报告。

#### Scenario: Full realtime workflow
- **WHEN** 用户选择动作类型后点击「开始跟做」
- **THEN** 系统 SHALL 显示 3 秒倒计时（3、2、1），然后开始实时分析，摄像头画面叠加骨骼，旁边面板实时刷新矫正建议

#### Scenario: Countdown display
- **WHEN** 倒计时进行中（3、2、1）
- **THEN** 系统 SHALL 在画面上方或状态栏显示倒计时数字，用户可利用此时间做好准备

#### Scenario: Realtime feedback display
- **WHEN** 实时分析过程中检测到关节角度偏差
- **THEN** 系统 SHALL 在旁边面板实时显示矫正建议（如「⚠️ 膝盖内扣，请外展」），并且正确的关节显示「✅ 位置正确」

#### Scenario: End session and show report
- **WHEN** 用户点击「结束」按钮
- **THEN** 系统 SHALL 停止实时分析，执行完整 DTW 对比，显示总体评分（0-100）、相似度百分比和总体矫正建议列表

## ADDED Requirements

### Requirement: Video analysis dual input
视频分析 Tab SHALL 支持两种输入方式：本地上传视频和摄像头录制视频，两种方式录入后走相同的分析流程。

#### Scenario: Upload local video
- **WHEN** 用户选择「上传视频」模式并上传本地视频文件
- **THEN** 系统 SHALL 对视频执行完整的矫正分析流程并显示报告

#### Scenario: Record via webcam
- **WHEN** 用户选择「摄像头录制」模式，完成录制
- **THEN** 系统 SHALL 对录制的骨骼序列执行完整的矫正分析流程并显示报告
