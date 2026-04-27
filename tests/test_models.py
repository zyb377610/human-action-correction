"""
深度学习模型模块单元测试

覆盖: 数据类型、骨骼图、三种模型前向传播、损失函数、早停、模型工厂
"""

import pytest
import torch
import numpy as np

from src.models.data_types import PredictionResult, ModelConfig
from src.models.skeleton_graph import get_adjacency_matrix, get_edge_list, get_normalized_adjacency
from src.models.base_model import BaseActionModel
from src.models.stgcn import STGCN
from src.models.bilstm import BiLSTMAttention
from src.models.transformer_model import TransformerClassifier
from src.models.model_factory import create_model, list_available_models
from src.models.trainer import MultiTaskLoss, EarlyStopping, set_seed


# ===== 数据类型测试 =====

class TestDataTypes:
    def test_prediction_result(self):
        result = PredictionResult(
            label="squat", confidence=0.95,
            quality_score=88.5,
            class_probs={"squat": 0.95, "lunge": 0.05},
        )
        assert result.label == "squat"
        assert result.confidence == 0.95
        assert result.quality_score == 88.5
        assert "squat" in result.class_probs

    def test_model_config_defaults(self):
        config = ModelConfig()
        assert config.backbone_type == "bilstm"
        assert config.num_classes == 5
        assert config.hidden_dim == 128

    def test_model_config_from_dict(self):
        d = {"backbone_type": "stgcn", "num_classes": 3, "unknown_key": "ignored"}
        config = ModelConfig.from_dict(d)
        assert config.backbone_type == "stgcn"
        assert config.num_classes == 3


# ===== 骨骼图测试 =====

class TestSkeletonGraph:
    def test_adjacency_matrix_shape(self):
        adj = get_adjacency_matrix()
        assert adj.shape == (33, 33)

    def test_adjacency_symmetric(self):
        adj = get_adjacency_matrix()
        np.testing.assert_array_equal(adj, adj.T)

    def test_self_loop(self):
        adj = get_adjacency_matrix(self_loop=True)
        assert np.all(np.diag(adj) == 1.0)

    def test_no_self_loop(self):
        adj = get_adjacency_matrix(self_loop=False)
        assert np.all(np.diag(adj) == 0.0)

    def test_edge_list_nonempty(self):
        edges = get_edge_list()
        assert len(edges) > 0
        assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)

    def test_normalized_adjacency(self):
        norm_adj = get_normalized_adjacency()
        assert norm_adj.shape == (33, 33)
        # 归一化后值应在 [0, 1] 范围
        assert norm_adj.min() >= -0.01
        assert norm_adj.max() <= 1.01


# ===== 模型前向传播测试 =====

class TestSTGCN:
    def test_forward_shape(self):
        model = STGCN(num_classes=5, in_channels=4, hidden_dim=32, num_layers=2)
        x = torch.randn(2, 60, 33, 4)  # (B, T, V, C)
        output = model(x)
        assert output["cls_logits"].shape == (2, 5)
        assert output["quality_score"].shape == (2, 1)

    def test_quality_score_range(self):
        model = STGCN(num_classes=5, in_channels=4, hidden_dim=32, num_layers=2)
        x = torch.randn(2, 60, 33, 4)
        output = model(x)
        q = output["quality_score"]
        assert q.min() >= 0.0
        assert q.max() <= 100.0


class TestBiLSTMAttention:
    def test_forward_shape(self):
        model = BiLSTMAttention(num_classes=5, input_dim=20, hidden_dim=64, num_layers=2)
        x = torch.randn(2, 60, 20)  # (B, T, D)
        output = model(x)
        assert output["cls_logits"].shape == (2, 5)
        assert output["quality_score"].shape == (2, 1)

    def test_attention_weights(self):
        model = BiLSTMAttention(num_classes=5, input_dim=20, hidden_dim=64)
        x = torch.randn(2, 60, 20)
        model(x)
        weights = model.get_attention_weights()
        assert weights is not None
        assert weights.shape == (2, 60)
        # 注意力权重和约为 1
        sums = weights.sum(dim=1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)

    def test_attention_with_mask(self):
        model = BiLSTMAttention(num_classes=5, input_dim=20, hidden_dim=64)
        x = torch.randn(2, 60, 20)
        mask = torch.ones(2, 60)
        mask[0, 30:] = 0  # 第一个样本后半部分填充
        output = model(x, mask)
        assert output["cls_logits"].shape == (2, 5)


class TestTransformerClassifier:
    def test_forward_shape(self):
        model = TransformerClassifier(
            num_classes=5, input_dim=20, hidden_dim=64,
            num_layers=2, num_heads=4,
        )
        x = torch.randn(2, 60, 20)
        output = model(x)
        assert output["cls_logits"].shape == (2, 5)
        assert output["quality_score"].shape == (2, 1)

    def test_position_encoding_effect(self):
        """不同帧顺序应产生不同输出"""
        model = TransformerClassifier(
            num_classes=5, input_dim=20, hidden_dim=64,
            num_layers=2, num_heads=4, dropout=0.0,
        )
        model.eval()
        x = torch.randn(1, 60, 20)
        x_reversed = x.flip(dims=[1])  # 反转帧顺序

        with torch.no_grad():
            out1 = model(x)["cls_logits"]
            out2 = model(x_reversed)["cls_logits"]

        # 输出应不同
        assert not torch.allclose(out1, out2, atol=1e-3)


# ===== 模型工厂测试 =====

class TestModelFactory:
    def test_create_stgcn(self):
        model = create_model("stgcn", num_classes=5, in_channels=4, hidden_dim=32)
        assert isinstance(model, STGCN)
        assert isinstance(model, BaseActionModel)

    def test_create_bilstm(self):
        model = create_model("bilstm", num_classes=5, input_dim=20)
        assert isinstance(model, BiLSTMAttention)

    def test_create_transformer(self):
        model = create_model("transformer", num_classes=5, input_dim=20)
        assert isinstance(model, TransformerClassifier)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="不支持的模型类型"):
            create_model("unknown")

    def test_list_models(self):
        models = list_available_models()
        assert "stgcn" in models
        assert "bilstm" in models
        assert "transformer" in models

    def test_create_from_config(self):
        config = ModelConfig(backbone_type="bilstm", num_classes=3, input_dim=20)
        model = create_model("bilstm", config=config)
        assert model.num_classes == 3


# ===== 损失函数测试 =====

class TestMultiTaskLoss:
    def test_loss_computation(self):
        loss_fn = MultiTaskLoss(lambda_quality=0.5)
        cls_logits = torch.randn(4, 5)
        quality_pred = torch.rand(4, 1) * 100
        labels = torch.randint(0, 5, (4,))
        quality_target = torch.rand(4) * 100

        losses = loss_fn(cls_logits, quality_pred, labels, quality_target)
        assert "total" in losses
        assert "cls" in losses
        assert "quality" in losses
        assert losses["total"].item() > 0

    def test_lambda_weight(self):
        """验证 lambda_quality 影响总损失"""
        cls_logits = torch.randn(4, 5)
        quality_pred = torch.rand(4, 1) * 100
        labels = torch.randint(0, 5, (4,))
        quality_target = torch.rand(4) * 100

        loss_fn1 = MultiTaskLoss(lambda_quality=0.0)
        loss_fn2 = MultiTaskLoss(lambda_quality=1.0)

        l1 = loss_fn1(cls_logits, quality_pred, labels, quality_target)
        l2 = loss_fn2(cls_logits, quality_pred, labels, quality_target)

        # lambda=0 时总损失等于分类损失
        assert abs(l1["total"].item() - l1["cls"].item()) < 1e-5


# ===== 早停测试 =====

class TestEarlyStopping:
    def test_no_early_stop(self):
        es = EarlyStopping(patience=3)
        assert es(0.5) is False
        assert es(0.6) is False
        assert es(0.7) is False
        assert es.early_stop is False

    def test_early_stop_triggered(self):
        es = EarlyStopping(patience=3)
        es(0.8)  # 最佳
        es(0.7)  # counter=1
        es(0.7)  # counter=2
        result = es(0.7)  # counter=3 -> 触发
        assert result is True
        assert es.early_stop is True

    def test_reset_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(0.5)
        es(0.4)  # counter=1
        es(0.4)  # counter=2
        es(0.6)  # 改善，重置
        assert es.counter == 0
        assert es.early_stop is False


# ===== 工具函数测试 =====

class TestUtils:
    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)