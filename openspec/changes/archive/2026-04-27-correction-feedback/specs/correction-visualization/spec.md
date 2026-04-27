## ADDED Requirements

### Requirement: 骨骼偏差标注图
系统 SHALL 在骨骼图上用颜色标注各关节的偏差程度（绿色=正常、黄色=轻微、红色=严重）。

#### Scenario: 绘制偏差标注骨骼图
- **WHEN** 调用 visualizer.draw_deviation_skeleton(report)
- **THEN** 输出带颜色标注的骨骼图，偏差大的关节用红色高亮

### Requirement: 文本报告渲染
系统 SHALL 支持将 CorrectionReport 渲染为格式化的文本图片或直接打印到控制台。

#### Scenario: 控制台打印报告
- **WHEN** 调用 visualizer.print_report(report)
- **THEN** 在控制台以格式化方式输出矫正报告

### Requirement: 偏差柱状图
系统 SHALL 生成各关节偏差的水平柱状图，直观展示哪些关节偏差最大。

#### Scenario: 绘制偏差柱状图
- **WHEN** 调用 visualizer.plot_deviation_bar(report)
- **THEN** 输出偏差柱状图 PNG，关节按偏差降序排列