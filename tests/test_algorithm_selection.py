"""
DTW 算法切换机制测试

测试覆盖：
- 四种模式（dtw / fastdtw / ddtw / auto）设置正确
- 自动模式解析为正确的实时/离线算法
- 无效算法名称抛出 ValueError
"""

import pytest

from src.app.pipeline import AppPipeline, ALGORITHM_CHOICES, ALGORITHM_DISPLAY_NAMES


class TestAlgorithmSelection:
    """算法选择机制测试"""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """创建 AppPipeline（使用临时模板目录）"""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        return AppPipeline(
            templates_dir=str(templates_dir),
            output_dir=str(tmp_path / "output"),
        )

    def test_default_algorithm(self, pipeline):
        """默认算法应为 auto"""
        assert pipeline.algorithm == "auto"

    def test_set_dtw(self, pipeline):
        """设置经典 DTW"""
        pipeline.set_algorithm("dtw")
        assert pipeline.algorithm == "dtw"
        assert pipeline._resolve_offline_algorithm() == "dtw"
        assert pipeline._resolve_realtime_algorithm() == "dtw"

    def test_set_fastdtw(self, pipeline):
        """设置 FastDTW"""
        pipeline.set_algorithm("fastdtw")
        assert pipeline.algorithm == "fastdtw"
        assert pipeline._resolve_offline_algorithm() == "fastdtw"
        assert pipeline._resolve_realtime_algorithm() == "fastdtw"

    def test_set_ddtw(self, pipeline):
        """设置 DerivativeDTW"""
        pipeline.set_algorithm("ddtw")
        assert pipeline.algorithm == "ddtw"
        assert pipeline._resolve_offline_algorithm() == "ddtw"
        assert pipeline._resolve_realtime_algorithm() == "ddtw"

    def test_set_auto(self, pipeline):
        """设置自动选择"""
        pipeline.set_algorithm("auto")
        assert pipeline.algorithm == "auto"
        # 自动模式：离线用 ddtw，实时用 fastdtw
        assert pipeline._resolve_offline_algorithm() == "ddtw"
        assert pipeline._resolve_realtime_algorithm() == "fastdtw"

    def test_invalid_algorithm(self, pipeline):
        """无效算法名称应抛出 ValueError"""
        with pytest.raises(ValueError):
            pipeline.set_algorithm("invalid")

    def test_algorithm_choices_defined(self):
        """确认算法选项列表已定义"""
        assert len(ALGORITHM_CHOICES) == 4
        labels = [label for label, _ in ALGORITHM_CHOICES]
        assert "经典 DTW" in labels
        assert "FastDTW" in labels
        assert "DerivativeDTW" in labels

    def test_algorithm_display_names(self):
        """确认显示名称映射完整"""
        assert "dtw" in ALGORITHM_DISPLAY_NAMES
        assert "fastdtw" in ALGORITHM_DISPLAY_NAMES
        assert "ddtw" in ALGORITHM_DISPLAY_NAMES
        assert "auto" in ALGORITHM_DISPLAY_NAMES

    def test_set_algorithm_updates_comparator(self, pipeline):
        """切换算法后应更新内部对比器"""
        pipeline.set_algorithm("fastdtw")
        assert pipeline._correction_pipeline._comparator._algorithm == "fastdtw"

        pipeline.set_algorithm("ddtw")
        assert pipeline._correction_pipeline._comparator._algorithm == "ddtw"

        pipeline.set_algorithm("auto")
        assert pipeline._correction_pipeline._comparator._algorithm == "ddtw"  # 离线用 ddtw
