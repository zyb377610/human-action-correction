## ADDED Requirements

### Requirement: AppPipeline 端到端编排
`AppPipeline` 类 SHALL 作为 UI 层与算法层的唯一桥梁，串联姿态估计 → 分类 → DTW 对比 → 矫正反馈的完整流程。

#### Scenario: 视频文件分析
- **WHEN** 调用 `analyze_video(video_path)` 并传入有效视频路径
- **THEN** 返回 `AnalysisResult`，包含 CorrectionReport、偏差柱状图路径、骨骼标注视频路径、摘要文本

#### Scenario: 指定动作类型分析
- **WHEN** 调用 `analyze_video(video_path, action_name="squat")`
- **THEN** 跳过自动分类，直接使用 "squat" 作为动作类型进行对比

#### Scenario: 骨骼序列直接分析
- **WHEN** 调用 `analyze_sequence(sequence)` 传入 numpy 数组
- **THEN** 跳过姿态估计步骤，直接从分类开始执行后续流程

### Requirement: 摄像头帧处理
`AppPipeline` SHALL 支持逐帧处理摄像头输入，返回带骨骼叠加的图像。

#### Scenario: 处理单帧
- **WHEN** 调用 `process_camera_frame(frame)` 传入 BGR 图像
- **THEN** 返回 `ProcessedFrame`，包含叠加骨骼的图像和当前帧的 landmarks

### Requirement: 模板录入
`AppPipeline` SHALL 支持从视频文件录入标准动作模板。

#### Scenario: 录入新模板
- **WHEN** 调用 `record_template(video_path, action_name)` 传入视频和动作名
- **THEN** 从视频提取骨骼序列，保存到模板库，返回 True

#### Scenario: 重复动作名录入
- **WHEN** 录入的动作名已存在模板
- **THEN** 新模板追加到该动作的模板列表中

### Requirement: AnalysisResult 数据结构
`AnalysisResult` SHALL 包含完整的分析结果，支持序列化和各 UI 组件所需的数据字段。

#### Scenario: 结果包含完整字段
- **WHEN** 分析完成
- **THEN** AnalysisResult 包含 action_name、quality_score、report_text、deviation_plot_path、skeleton_video_path、corrections 列表
