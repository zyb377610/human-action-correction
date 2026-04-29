## 1. 实时反馈引擎

- [ ] 1.1 创建 `src/correction/realtime_feedback.py`，定义 `FeedbackItem`（关节名称、偏差方向、矫正提示、严重等级）和 `FeedbackSnapshot`（建议列表、has_pose、窗口相似度、时间戳）数据类
- [ ] 1.2 实现 `RealtimeFeedbackEngine` 类：初始化时接收模板 PoseSequence、算法名称、窗口大小（默认 10）、角度偏差阈值（默认 15 度）
- [ ] 1.3 实现 `analyze_frame()` 方法：接收当前帧 landmarks + 当前帧索引 + 预期总帧数 → 缓存到帧窗口 → 映射模板对应帧段 → 固定窗口 DTW 对比 → 角度偏差检测 → 规则引擎建议 → 返回 FeedbackSnapshot。验证：单次调用耗时 ≤10ms
- [ ] 1.4 实现模板进度映射与窗口提取逻辑：user_progress → template 中心帧 → 取 ±half_window 范围，含越界保护
- [ ] 1.5 实现建议去重逻辑：同一建议在 2 秒内不重复，使用时间戳缓存

## 2. DTW 算法选择机制

- [ ] 2.1 在 `src/app/pipeline.py` 的 `AppPipeline` 中新增 `set_algorithm()` 方法，支持设置当前使用的 DTW 算法（dtw / fastdtw / ddtw / auto），动态传递给 `ActionComparator` 和 `RealtimeFeedbackEngine`
- [ ] 2.2 实现「自动选择」逻辑：实时窗口用 fastdtw，离线完整序列用 ddtw
- [ ] 2.3 在分析报告（AnalysisResult）中新增 algorithm_used 字段，报告文本中显示「使用算法: xxx」

## 3. 会话状态管理增强

- [ ] 3.1 在 `src/app/session.py` 中新增实时模式状态：COUNTDOWN（倒计时）、REALTIME（实时分析中），扩展 RecordingState 枚举
- [ ] 3.2 新增 `start_countdown()` 方法（启动 3 秒倒计时）、`start_realtime()` 方法（倒计时结束后进入实时分析）、`stop_realtime()` 方法（结束实时分析，返回完整帧序列）
- [ ] 3.3 新增实时反馈缓存：存储最新的 FeedbackSnapshot，供 UI 轮询获取

## 4. Pipeline 层增强

- [ ] 4.1 在 `src/app/pipeline.py` 中新增 `init_realtime_session()` 方法：根据动作名称加载模板序列，创建 RealtimeFeedbackEngine 实例
- [ ] 4.2 新增 `process_realtime_frame()` 方法：接收 numpy 帧 → PoseEstimator → RealtimeFeedbackEngine.analyze_frame() → 返回 ProcessedFrame + FeedbackSnapshot
- [ ] 4.3 在 `src/app/data_types.py` 中扩展 ProcessedFrame，新增 feedback 字段（可选 FeedbackSnapshot）

## 5. UI 重构 — 实时模式 Tab

- [ ] 5.1 重写实时模式 Tab 布局：左侧摄像头画面（骨骼叠加），右侧实时矫正建议面板（Markdown 实时刷新），底部动作选择 + 算法选择 + 开始/结束按钮，下方总体报告区域
- [ ] 5.2 实现 3 秒倒计时逻辑：点击「开始跟做」后，状态栏显示 3、2、1 倒计时，倒计时结束后自动进入实时分析
- [ ] 5.3 实现实时 stream 回调：每帧调用 process_realtime_frame()，输出骨骼画面到 camera_display，输出 FeedbackSnapshot 格式化为 Markdown 到建议面板
- [ ] 5.4 实现「结束」回调：停止实时分析 → 调用 analyze_sequence() 完整 DTW → 显示总体评分 + 总体报告（含使用的算法名称）

## 6. UI 增强 — 视频分析 Tab

- [ ] 6.1 在视频分析 Tab 新增输入模式切换（Radio：「上传视频」/「摄像头录制」），根据选择显示/隐藏对应组件
- [ ] 6.2 新增算法选择下拉框（DTW / FastDTW / DDTW / 自动选择），分析时传递给 pipeline
- [ ] 6.3 实现摄像头录制子模式：摄像头画面 + 开始/停止录制按钮，录制完成后自动调用 analyze_sequence() 走标准分析流程

## 7. 测试与验证

- [ ] 7.1 为 RealtimeFeedbackEngine 编写单元测试：测试固定窗口 DTW 正确性、阈值过滤、去重逻辑、进度映射越界保护
- [ ] 7.2 为算法切换机制编写测试：验证四种模式下 pipeline 正确调用对应算法
- [ ] 7.3 运行全量测试，确保原有 164 个测试不受影响
- [ ] 7.4 手动端到端测试：启动 Web 界面，验证实时模式完整流程（倒计时→跟做→实时建议→总体报告）、视频分析摄像头录制模式、算法切换功能
