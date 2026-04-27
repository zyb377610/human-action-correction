"""
模型工厂函数

根据模型类型名称创建对应的模型实例，统一模型创建入口。
"""

from typing import Optional

from .base_model import BaseActionModel
from .data_types import ModelConfig
from .stgcn import STGCN
from .bilstm import BiLSTMAttention
from .transformer_model import TransformerClassifier


# 支持的模型类型注册表
MODEL_REGISTRY = {
    "stgcn": STGCN,
    "bilstm": BiLSTMAttention,
    "transformer": TransformerClassifier,
}


def create_model(
    model_type: str,
    num_classes: int = 5,
    config: Optional[ModelConfig] = None,
    **kwargs,
) -> BaseActionModel:
    """
    根据模型类型创建对应模型实例

    Args:
        model_type: 模型类型 ("stgcn" / "bilstm" / "transformer")
        num_classes: 动作类别数
        config: ModelConfig 配置对象（优先级高于 kwargs）
        **kwargs: 传递给模型构造函数的额外参数

    Returns:
        BaseActionModel 子类实例

    Raises:
        ValueError: 不支持的模型类型
    """
    model_type = model_type.lower().strip()

    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"不支持的模型类型: '{model_type}'。支持: {supported}"
        )

    # 从 config 提取参数
    if config is not None:
        params = {
            "num_classes": config.num_classes,
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "in_channels": config.in_channels,
        }
        params.update(kwargs)  # kwargs 可覆盖
    else:
        params = {"num_classes": num_classes, **kwargs}

    model_cls = MODEL_REGISTRY[model_type]
    return model_cls(**params)


def list_available_models():
    """列出所有支持的模型类型"""
    return list(MODEL_REGISTRY.keys())