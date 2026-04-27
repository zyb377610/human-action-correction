## 1. 统一配置管理

- [x] 1.1 创建 `configs/default_config.yaml` — 统一配置文件，包含 paths（templates_dir, models_dir, output_dir）、model（type, checkpoint）、pose_estimation（model_path, max_frames）、server（port, share）。验证：YAML 格式正确可解析
- [x] 1.2 实现 `src/utils/config.py` — load_config(path=None) 函数加载配置文件，支持默认路径回退、环境变量覆盖。验证：load_config() 返回正确的字典

## 2. 端到端集成测试

- [x] 2.1 编写 `tests/test_integration.py` — 集成测试：模块间数据流测试（PoseSequence → 预处理 → DTW → 规则引擎 → CorrectionReport），验证各模块输出格式匹配下游输入。验证：pytest 通过
- [x] 2.2 集成测试：SessionManager 全生命周期测试 — 从 idle → recording → add_frames → stop → analyzing → finish 完整状态机。验证：pytest 通过
- [x] 2.3 运行全量测试套件 — 执行 `pytest tests/ -v` 确保所有模块的单元测试 + 集成测试全部通过（0 failure）。验证：164 passed, 0 failed

## 3. 离线批处理

- [x] 3.1 编写 `scripts/batch_process.py` — 批处理脚本：输入视频文件夹 → 逐个分析 → 输出 CSV 汇总（文件名, 动作, 评分, 矫正数）+ 每个视频的 txt 报告。验证：脚本可正常运行

## 4. 演示准备

- [x] 4.1 编写 `scripts/prepare_demo.py` — 演示准备脚本：使用合成数据为 5 种动作各生成一个模板，保存到模板库。验证：运行后 data/templates 下有 5 个动作目录

## 5. 文档更新

- [x] 5.1 更新 `README.md` — 更新快速开始（实际启动命令）、功能介绍（各阶段技术）、开发计划打钩、项目结构完善。验证：内容完整准确
- [x] 5.2 更新 `docs/design/roadmap.md` — 标记 Phase 1-7 全部完成状态。验证：所有 Phase 标记为 ✅