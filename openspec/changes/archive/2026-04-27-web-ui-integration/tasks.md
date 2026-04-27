## 1. 数据类型与应用层流水线

- [x] 1.1 实现 `src/app/data_types.py` — 定义 AnalysisResult（action_name, quality_score, report_text, deviation_plot_path, skeleton_video_path, corrections, report 对象）和 ProcessedFrame（annotated_image, landmarks）数据类。验证：可实例化并访问所有字段
- [x] 1.2 实现 `src/app/pipeline.py` — AppPipeline 类：analyze_video(video_path, action_name=None, progress_callback=None) 串联 PoseEstimator → Predictor → Comparator → CorrectionPipeline → ReportVisualizer，返回 AnalysisResult。验证：传入测试视频返回完整结果
- [x] 1.3 实现 AppPipeline.analyze_sequence() — 接收 numpy 数组直接分析，跳过姿态估计。验证：传入 (T,33,4) 数组返回 AnalysisResult
- [x] 1.4 实现 AppPipeline.process_camera_frame() — 单帧处理，返回带骨骼叠加的图像和 landmarks。验证：传入 BGR 图像返回 ProcessedFrame
- [x] 1.5 实现 AppPipeline.record_template() — 从视频提取骨骼序列并保存到模板库。验证：录入后模板库新增一条记录

## 2. 会话状态管理

- [x] 2.1 实现 `src/app/session.py` — SessionManager 类：管理录制状态（idle/recording/analyzing）、帧缓存（landmarks 列表）、最近分析结果缓存、模板列表刷新。验证：状态机转换正确
- [x] 2.2 实现 SessionManager 模板列表管理 — 启动时扫描模板库加载可用动作列表，录入新模板后自动刷新。验证：get_action_list() 返回正确的动作名列表

## 3. CorrectionPipeline 进度回调适配

- [x] 3.1 修改 `src/correction/pipeline.py` — CorrectionPipeline.analyze() 增加 progress_callback 可选参数，在分类、对比、偏差分析、反馈生成各阶段调用 callback(step, total, message)。验证：传入回调函数后收到 4 次调用

## 4. Gradio Web 界面

- [x] 4.1 实现 `src/app/gradio_ui.py` — 主界面框架：gr.Blocks + 4 个 gr.Tab（视频分析、实时模式、模板管理、系统说明），配置中文标题和主题。验证：启动后浏览器显示 4 个 Tab
- [x] 4.2 实现视频分析 Tab — 视频上传组件 + 动作类型下拉框（自动识别 + 各动作选项）+ 分析按钮 → 报告文本 + 偏差柱状图 + 骨骼标注视频 + 进度条。验证：上传视频后展示完整分析结果
- [x] 4.3 实现实时摄像头 Tab — 摄像头流组件（实时骨骼叠加）+ 开始录制/停止分析按钮 + 状态提示 + 分析结果展示区。验证：摄像头画面带骨骼标注，录制后输出报告
- [x] 4.4 实现模板管理 Tab — 模板列表展示（动作名、帧数、时间）+ 视频上传录入新模板 + 动作名输入框 + 录入按钮。验证：查看已有模板并成功录入新模板
- [x] 4.5 实现系统说明 Tab — Markdown 渲染系统简介、使用指南、技术架构说明（含流程图描述）。验证：内容完整显示

## 5. 启动脚本与依赖

- [x] 5.1 编写 `scripts/launch_app.py` — 一键启动脚本：初始化 AppPipeline、创建 Gradio UI、launch(server_name="0.0.0.0", share=False)。验证：python scripts/launch_app.py 可启动 Web 服务
- [x] 5.2 更新 `requirements.txt` — 添加 gradio>=4.0 依赖。验证：pip install -r requirements.txt 可安装所有依赖

## 6. 模块导出与测试

- [x] 6.1 更新 `src/app/__init__.py` — 导出 AppPipeline、SessionManager、AnalysisResult、create_gradio_app。验证：from src.app import AppPipeline 可正常导入
- [x] 6.2 编写 `tests/test_app.py` — 单元测试覆盖 AppPipeline（视频分析、序列分析、帧处理）、SessionManager（状态转换、模板列表）。验证：所有测试通过