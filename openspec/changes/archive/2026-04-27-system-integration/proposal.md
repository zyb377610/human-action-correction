## Why

Phase 1-6 已完成所有核心算法模块和 Web 界面，但各模块之间缺乏端到端的集成测试验证，README 尚未更新以反映实际开发成果，离线批处理模式尚未实现，整体系统也缺少一个完整的演示准备（示例数据、一键脚本等）。作为毕业设计的收尾阶段，Phase 7 需要确保系统可靠、文档完善、演示流畅。

## What Changes

- 新增 `tests/test_integration.py` — 端到端集成测试，覆盖完整的视频 → 矫正报告流水线
- 新增 `scripts/batch_process.py` — 离线批处理脚本，支持批量处理视频文件夹并输出报告
- 新增 `scripts/prepare_demo.py` — 演示准备脚本，自动录入示例模板数据
- 更新 `README.md` — 反映实际项目成果、启动指令、开发计划打钩
- 更新 `docs/design/roadmap.md` — 标记所有 Phase 完成
- 新增 `configs/default_config.yaml` — 统一系统配置文件（模型路径、模板目录、服务端口等）
- 整合所有现有测试，确保全量通过

## Capabilities

### New Capabilities
- `integration-test`: 端到端集成测试套件，使用 mock 数据模拟完整流水线
- `batch-processing`: 离线批处理模式，支持从文件夹批量处理视频并导出 CSV 汇总报告
- `system-config`: 统一配置管理，所有模块共用 default_config.yaml 中的路径和参数

### Modified Capabilities
- `readme-docs`: README.md 更新为完整的项目文档，包含实际启动指令和功能描述

## Impact

- **新增文件**: `tests/test_integration.py`、`scripts/batch_process.py`、`scripts/prepare_demo.py`、`configs/default_config.yaml`
- **修改文件**: `README.md`、`docs/design/roadmap.md`
- **无新依赖**
- **预计开发时间**: 0.5-1 天
