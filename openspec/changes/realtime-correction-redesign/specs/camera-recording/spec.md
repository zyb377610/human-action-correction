## ADDED Requirements

### Requirement: Camera recording in video analysis
视频分析 Tab SHALL 支持「摄像头录制」输入方式，用户可选择通过摄像头录制动作视频，录制完成后走与本地上传视频相同的分析流程。

#### Scenario: User records via webcam
- **WHEN** 用户在视频分析 Tab 选择「摄像头录制」模式，点击开始录制并执行动作后点击停止
- **THEN** 系统 SHALL 将录制的帧序列保存为临时视频文件或直接提取骨骼序列，然后调用 analyze_sequence 进行分析

#### Scenario: Recording with insufficient frames
- **WHEN** 用户录制帧数不足 5 帧
- **THEN** 系统 SHALL 提示「录制帧数不足，请确保摄像头能看到全身」

### Requirement: Input mode switching
视频分析 Tab SHALL 提供输入模式切换控件，用户可在「上传视频」和「摄像头录制」之间切换。

#### Scenario: Switch to upload mode
- **WHEN** 用户选择「上传视频」模式
- **THEN** 系统 SHALL 显示文件上传组件，隐藏摄像头组件

#### Scenario: Switch to recording mode
- **WHEN** 用户选择「摄像头录制」模式
- **THEN** 系统 SHALL 显示摄像头组件和录制控制按钮，隐藏文件上传组件
