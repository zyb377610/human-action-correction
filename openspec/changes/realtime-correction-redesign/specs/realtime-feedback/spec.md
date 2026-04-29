## ADDED Requirements

### Requirement: Fixed-window DTW realtime feedback
系统 SHALL 对每一帧摄像头画面进行固定窗口 DTW 分析：缓存最近 10 帧 landmarks，与标准模板对应进度的 10 帧做 DTW 对比，结合角度偏差生成即时矫正建议。

#### Scenario: Normal frame with detectable pose and sufficient buffer
- **WHEN** 摄像头帧被传入 RealtimeFeedbackEngine 且已缓存 ≥10 帧有效 landmarks
- **THEN** 系统 SHALL 在 300ms 内返回包含窗口 DTW 相似度、关节角度偏差和矫正建议文字列表的 FeedbackSnapshot

#### Scenario: Insufficient buffer frames
- **WHEN** 已缓存帧数 < 10（刚开始录制）
- **THEN** 系统 SHALL 使用已有帧做缩小窗口的 DTW 对比，或仅做单帧角度对比作为降级方案

#### Scenario: Frame without detectable pose
- **WHEN** 摄像头帧中未检测到人体姿态
- **THEN** 系统 SHALL 返回空建议列表，并在 FeedbackSnapshot 中标记 has_pose=False

### Requirement: Template progress mapping
系统 SHALL 基于时间比例将用户当前进度映射到标准模板的对应帧位置，取 ±5 帧范围作为窗口对比的模板帧段。

#### Scenario: User at 50% progress
- **WHEN** 用户已完成预期帧数的 50%
- **THEN** 系统 SHALL 使用模板序列中第 50% 位置为中心的 10 帧作为参考窗口

#### Scenario: User exceeds expected duration
- **WHEN** 用户当前帧数超过预期总帧数
- **THEN** 系统 SHALL 使用模板最后 10 帧作为参考窗口，不发生越界错误

### Requirement: Angle deviation threshold
系统 SHALL 仅在关节角度偏差超过阈值（默认 15 度）时生成矫正建议，避免噪声级别的微小偏差触发提示。

#### Scenario: Minor deviation below threshold
- **WHEN** 用户某关节角度与模板偏差为 10 度（< 15 度阈值）
- **THEN** 系统 SHALL 不对该关节生成矫正建议

#### Scenario: Significant deviation above threshold
- **WHEN** 用户某关节角度与模板偏差为 25 度（> 15 度阈值）
- **THEN** 系统 SHALL 对该关节生成矫正建议，包含关节名称、偏差方向和矫正提示文字

### Requirement: Feedback deduplication
系统 SHALL 对连续帧中重复的建议进行去重，同一建议在 2 秒内不重复显示。

#### Scenario: Same advice in consecutive frames
- **WHEN** 连续 30 帧（1 秒）都检测到「膝盖内扣」
- **THEN** 系统 SHALL 仅显示一次「膝盖内扣」建议，直到该建议消失后 2 秒才可再次显示

### Requirement: Post-session DTW analysis
录制结束后，系统 SHALL 使用完整的帧序列进行 DTW 对比分析，生成总体评分和总体矫正报告。

#### Scenario: Recording completed with sufficient frames
- **WHEN** 用户点击「结束」且录制帧数 ≥ 5
- **THEN** 系统 SHALL 执行完整的 CorrectionPipeline 分析，返回包含 quality_score、similarity、使用的算法名称和 corrections 列表的 AnalysisResult
