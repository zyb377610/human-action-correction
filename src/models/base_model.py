"""
动作分类模型抽象基类

定义所有动作分类模型的统一接口，保证输出格式一致。
所有模型 MUST 继承 BaseActionModel 并实现 forward() 方法。
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseActionModel(ABC, nn.Module):
    """
    动作分类模型抽象基类

    所有模型输出为 dict:
        - "cls_logits": (B, num_classes) 分类 logits
        - "quality_score": (B, 1) 质量评分 [0, 100]
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列张量
               - keypoints 模式: (B, T, 33, 4)
               - features 模式: (B, T, D)
            mask: padding mask (B, T)，1=有效帧，0=填充帧

        Returns:
            dict 含:
                - "cls_logits": (B, num_classes)
                - "quality_score": (B, 1)，值域 [0, 100]
        """
        ...

    def get_num_parameters(self) -> int:
        """获取模型总参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """获取可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)