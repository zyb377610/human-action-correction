## ADDED Requirements

### Requirement: 端到端集成测试
系统 SHALL 提供集成测试套件，使用合成数据验证从骨骼序列输入到矫正报告输出的完整流水线。

#### Scenario: 合成数据完整流水线
- **WHEN** 传入合成的 (T, 33, 4) numpy 骨骼序列和动作名称
- **THEN** AppPipeline.analyze_sequence() 返回包含 report_text、quality_score 的 AnalysisResult

#### Scenario: 模块间数据流验证
- **WHEN** 逐步调用各模块（预处理 → DTW 对比 → 规则引擎 → 反馈生成）
- **THEN** 每个模块的输出格式与下游模块的输入格式匹配

### Requirement: 离线批处理
系统 SHALL 支持批量处理视频文件夹，输出汇总 CSV 和逐个视频的详细报告。

#### Scenario: 批处理视频文件夹
- **WHEN** 运行 batch_process.py 并指定包含多个视频的文件夹
- **THEN** 为每个视频生成矫正报告，并输出一个 CSV 汇总文件

### Requirement: 统一配置管理
系统 SHALL 提供 YAML 配置文件，集中管理模板目录、模型路径、服务端口等参数。

#### Scenario: 加载默认配置
- **WHEN** 未指定自定义配置文件
- **THEN** 系统使用 configs/default_config.yaml 中的默认值
