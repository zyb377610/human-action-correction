## 1. 数据类型与角度工具

- [x] 1.1 实现 `src/correction/data_types.py` — 定义 CorrectionItem（关节名、偏差描述、矫正建议、优先级）、CorrectionReport（动作类别、质量评分、整体评语、建议列表、详情数据），提供 to_text() 和 to_dict() 方法。验证：CorrectionReport 可实例化并序列化
- [x] 1.2 实现 `src/correction/angle_utils.py` — AngleCalculator 类：基于三点向量夹角计算关节角度，定义关键角度映射表（膝角/肘角/髋角/肩角等），提供 compute_angles(PoseSequence) 和 compare_angles(user, template, path) 方法。验证：计算已知坐标的角度值与预期一致

## 2. 矫正规则引擎

- [x] 2.1 实现 `src/correction/rules.py` — CorrectionRule dataclass（触发条件、建议模板、优先级）和 CorrectionRuleEngine 类（规则注册、规则匹配）。验证：注册规则后 match_rules() 能正确匹配并输出 CorrectionItem
- [x] 2.2 实现 5 类动作的内置规则集 — 为 squat/arm_raise/side_bend/lunge/standing_stretch 各定义 8-12 条矫正规则，覆盖主要关节偏差场景。验证：每个动作类别至少有 8 条规则
- [x] 2.3 实现通用兜底规则 — 当关节偏差超过阈值但无专属规则时，生成通用建议。验证：未注册规则的关节偏差也能产生兜底建议

## 3. 反馈报告生成

- [x] 3.1 实现 `src/correction/feedback.py` — FeedbackGenerator 类：接收 DeviationReport + AngleDeviations + PredictionResult，生成 CorrectionReport（含整体评语 + 排序的建议列表）。验证：给定偏差数据输出完整报告
- [x] 3.2 实现整体评语生成逻辑 — 根据质量评分分级（≥90 优秀 / 70-89 良好 / 50-69 需改进 / <50 较差），生成对应中文评语。验证：不同分数段生成不同评语
- [x] 3.3 实现建议优先级排序 — 按优先级（high/medium/low）排序，同级内按偏差程度降序。验证：输出列表的顺序正确

## 4. 端到端流水线

- [x] 4.1 实现 `src/correction/pipeline.py` — CorrectionPipeline 类：串联 ActionPredictor + ActionComparator + JointDeviationAnalyzer + AngleCalculator + CorrectionRuleEngine + FeedbackGenerator。验证：pipeline.analyze(PoseSequence) 返回 CorrectionReport
- [x] 4.2 实现指定模式 — analyze(pose_sequence, action_name="squat") 跳过分类直接对比。验证：手动指定动作类别时不调用 ActionPredictor
- [x] 4.3 实现错误处理 — 模板库为空、checkpoint 不存在等异常给出明确提示。验证：无模板时抛出 ValueError 含引导信息

## 5. 矫正可视化

- [x] 5.1 实现 `src/correction/report_visualizer.py` — ReportVisualizer 类：print_report() 控制台输出 + draw_deviation_skeleton() 骨骼偏差标注图 + plot_deviation_bar() 偏差柱状图。验证：生成偏差柱状图 PNG 文件

## 6. 模块集成与测试

- [x] 6.1 更新 `src/correction/__init__.py` — 导出所有公共 API（CorrectionPipeline, CorrectionReport, CorrectionItem, FeedbackGenerator, CorrectionRuleEngine, AngleCalculator, ReportVisualizer）。验证：from src.correction import CorrectionPipeline 可正常导入
- [x] 6.2 编写 `tests/test_correction.py` — 单元测试覆盖角度计算、规则匹配、评语生成、报告序列化。验证：所有测试通过

## 7. 脚本与端到端演示

- [x] 7.1 编写 `scripts/run_correction.py` — 端到端演示脚本：加载序列文件 → 矫正分析 → 打印报告 + 保存可视化图。验证：脚本运行后输出矫正报告文本和偏差图