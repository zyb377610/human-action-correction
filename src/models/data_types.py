"""
深度学习模型数据类型定义

定义模型模块的核心数据结构：
- PredictionResult: 模型预测结果（分类标签 + 置信度 + 质量评分）
- ModelConfig: 模型配置参数
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PredictionResult:
    """
    模型预测结果

    Attributes:
        label: 预测的动作类别名称（如 "squat"）
        confidence: 预测置信度 [0, 1]
        quality_score: 动作质量评分 [0, 100]
        class_probs: 各类别的预测概率 {类别名: 概率}
    """
    label: str
    confidence: float
    quality_score: float
    class_probs: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PredictionResult(label='{self.label}', "
            f"confidence={self.confidence:.3f}, "
            f"quality_score={self.quality_score:.1f})"
        )


@dataclass
class ModelConfig:
    """
    模型配置参数

    Attributes:
        backbone_type: 模型架构类型 ("stgcn" / "bilstm" / "transformer")
        num_classes: 动作类别数
        input_dim: 输入特征维度（特征模式下使用）
        hidden_dim: 隐藏层维度
        num_layers: 网络层数
        num_heads: 注意力头数（Transformer 使用）
        dropout: Dropout 比率
        num_joints: 关键点数量（ST-GCN 使用）
        in_channels: 输入通道数（ST-GCN 使用，x/y/z/visibility = 4）
        target_frames: 序列统一长度
        learning_rate: 学习率
        weight_decay: 权重衰减
        batch_size: 批大小
        epochs: 最大训练轮数
        patience: 早停耐心值
        lambda_quality: 质量评分损失权重
        seed: 随机种子
    """
    backbone_type: str = "bilstm"
    num_classes: int = 5
    input_dim: int = 20
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3
    num_joints: int = 33
    in_channels: int = 4
    target_frames: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    lambda_quality: float = 0.5
    seed: int = 42

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        """从配置字典创建 ModelConfig（忽略未知键）"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)