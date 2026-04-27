"""
BiLSTM-Attention 时序分类模型

双层双向 LSTM + Bahdanau 注意力机制，处理特征向量序列 (B, T, D)。
参考 ASD 综述中 CNN-LSTM 时空特征方案。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseActionModel


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) 注意力机制

    计算序列中每一帧的注意力权重，生成加权上下文向量。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            hidden_states: (B, T, H) LSTM 输出序列
            mask: (B, T) padding mask，1=有效，0=填充

        Returns:
            (context, weights):
                context: (B, H) 加权上下文向量
                weights: (B, T) 注意力权重
        """
        # 注意力分数: (B, T, H) -> (B, T, 1) -> (B, T)
        energy = torch.tanh(self.W(hidden_states))
        scores = self.v(energy).squeeze(-1)  # (B, T)

        # 掩码处理：将填充位置的分数设为 -inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, T)

        # 加权求和: (B, 1, T) @ (B, T, H) -> (B, 1, H) -> (B, H)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)

        return context, weights


class BiLSTMAttention(BaseActionModel):
    """
    BiLSTM + Bahdanau 注意力分类模型

    架构: 输入投影 -> 双层 BiLSTM -> 注意力聚合 -> 双头输出
    输入: (B, T, D) 特征向量序列
    输出: cls_logits (B, num_classes), quality_score (B, 1)
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # BiLSTM 输出维度 = 2 * hidden_dim
        lstm_out_dim = hidden_dim * 2

        self.attention = BahdanauAttention(lstm_out_dim)

        # 双头输出
        self.cls_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 保存最近一次注意力权重
        self._attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 特征向量序列
            mask: (B, T) padding mask

        Returns:
            {"cls_logits": (B, num_classes), "quality_score": (B, 1)}
        """
        # 输入投影
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden_dim)

        # 注意力聚合
        context, weights = self.attention(lstm_out, mask)  # (B, 2*hidden_dim)
        self._attention_weights = weights.detach()

        # 双头输出
        cls_logits = self.cls_head(context)
        quality_score = self.quality_head(context) * 100.0

        return {
            "cls_logits": cls_logits,
            "quality_score": quality_score,
        }

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        获取最近一次前向传播的注意力权重

        Returns:
            (B, T) 注意力权重，值之和为 1；未执行前向传播时返回 None
        """
        return self._attention_weights