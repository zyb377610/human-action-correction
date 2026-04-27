"""
深度学习模型模块

提供动作分类和质量评估能力：
- 三种模型架构: ST-GCN, BiLSTM-Attention, Transformer
- 数据集构建和训练流水线
- 统一的推理接口
"""

from .data_types import PredictionResult, ModelConfig
from .base_model import BaseActionModel
from .stgcn import STGCN
from .bilstm import BiLSTMAttention
from .transformer_model import TransformerClassifier
from .model_factory import create_model, list_available_models
from .dataset import ActionDataset, create_dataloaders, split_dataset
from .trainer import Trainer, MultiTaskLoss, EarlyStopping, set_seed, get_device
from .predictor import ActionPredictor
from .skeleton_graph import get_adjacency_matrix, get_edge_list

__all__ = [
    # 数据类型
    "PredictionResult",
    "ModelConfig",
    # 模型
    "BaseActionModel",
    "STGCN",
    "BiLSTMAttention",
    "TransformerClassifier",
    "create_model",
    "list_available_models",
    # 数据集
    "ActionDataset",
    "create_dataloaders",
    "split_dataset",
    # 训练
    "Trainer",
    "MultiTaskLoss",
    "EarlyStopping",
    "set_seed",
    "get_device",
    # 推理
    "ActionPredictor",
    # 骨骼图
    "get_adjacency_matrix",
    "get_edge_list",
]