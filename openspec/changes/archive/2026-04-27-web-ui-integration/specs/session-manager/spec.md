## ADDED Requirements

### Requirement: 摄像头录制状态管理
`SessionManager` SHALL 管理摄像头录制过程中的状态（空闲/录制中/分析中），维护帧序列缓存。

#### Scenario: 开始录制
- **WHEN** 用户点击"开始录制"
- **THEN** 状态切换为"录制中"，开始缓存每帧的 landmarks 数据

#### Scenario: 停止录制
- **WHEN** 用户点击"停止并分析"
- **THEN** 状态切换为"分析中"，将缓存的帧序列提交给 AppPipeline 分析

#### Scenario: 分析完成后重置
- **WHEN** 分析完成并展示报告
- **THEN** 状态切换回"空闲"，清空帧缓存

### Requirement: 分析结果缓存
`SessionManager` SHALL 缓存最近一次分析结果，避免重复计算。

#### Scenario: 缓存命中
- **WHEN** 用户在结果展示区切换查看报告文本和偏差图
- **THEN** 直接使用缓存结果，无需重新计算

### Requirement: 模板加载状态
`SessionManager` SHALL 在应用启动时预加载模板库信息，并在模板变更时自动刷新。

#### Scenario: 启动加载模板
- **WHEN** 应用启动
- **THEN** 自动扫描模板库目录，加载可用动作类型列表

#### Scenario: 新模板录入后刷新
- **WHEN** 用户成功录入新模板
- **THEN** 动作类型下拉框自动更新，包含新录入的动作
