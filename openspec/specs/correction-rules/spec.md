# correction-rules Specification

## Purpose
TBD - created by archiving change correction-feedback. Update Purpose after archive.
## Requirements
### Requirement: 矫正规则定义
系统 SHALL 支持为每个动作类别定义矫正规则集，每条规则包含触发条件、建议文本模板和优先级。

#### Scenario: 规则匹配
- **WHEN** 深蹲动作的左膝关节角度偏差 > 10°
- **THEN** 规则引擎匹配到对应规则，输出 CorrectionItem("左膝弯曲角度不够，建议再下蹲约{angle_diff:.0f}°")

### Requirement: 通用兜底规则
系统 SHALL 提供通用兜底规则，当某关节偏差超过阈值但无专属规则时，自动生成通用建议。

#### Scenario: 无专属规则时兜底
- **WHEN** 某关节偏差 > moderate 阈值且无动作专属规则
- **THEN** 系统生成通用建议"该部位动作与标准存在偏差，请注意调整"

### Requirement: 规则可扩展注册
系统 SHALL 提供 register_rule() 和 register_action_rules() 接口，支持动态添加规则。

#### Scenario: 注册自定义规则
- **WHEN** 用户调用 engine.register_rule(action, rule)
- **THEN** 新规则加入引擎，后续匹配时生效

### Requirement: 文本模板变量插值
规则的建议文本 SHALL 支持变量插值，包括 {joint_name}、{angle_diff}、{deviation} 等变量。

#### Scenario: 角度差插值
- **WHEN** 规则模板为"建议调整约{angle_diff:.0f}°"，角度差为 12.3°
- **THEN** 输出建议文本为"建议调整约12°"

