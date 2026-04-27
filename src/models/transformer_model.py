"""
Transformer Encoder 序列分类模型

基于位置编码 + 多头自注意力的序列分类器。
参考 ViT 思路适配 1D 关键点特征序列。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base_model import BaseActionModel


class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    为序列中的每个位置生成唯一的位置向量，
    使 Transformer 能感知帧的顺序信息。
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) 添加位置编码后的序列
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(BaseActionModel):
    """
    Transformer Encoder 分类模型

    架构: 输入投影 -> 位置编码 -> N 层 Encoder -> [CLS] 池化 -> 双头输出
    输入: (B, T, D) 特征向量序列
    输出: cls_logits (B, num_classes), quality_score (B, 1)
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        # 输入投影到 hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # [CLS] token（可学习）
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=500, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # 双头输出
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 特征向量序列
            mask: (B, T) padding mask，1=有效，0=填充

        Returns:
            {"cls_logits": (B, num_classes), "quality_score": (B, 1)}
        """
        B, T, D = x.shape

        # 输入投影
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # 拼接 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_dim)

        # 位置编码
        x = self.pos_encoding(x)

        # 构建 attention mask（[CLS] 始终可见）
        src_key_padding_mask = None
        if mask is not None:
            # 为 [CLS] 添加 mask（始终有效 = False in PyTorch convention）
            cls_mask = torch.ones(B, 1, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)
            # PyTorch TransformerEncoder: True = 忽略该位置
            src_key_padding_mask = (full_mask == 0)

        # Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # 取 [CLS] token 的输出
        cls_output = x[:, 0, :]  # (B, hidden_dim)

        # 双头输出
        cls_logits = self.cls_head(cls_output)
        quality_score = self.quality_head(cls_output) * 100.0

        return {
            "cls_logits": cls_logits,
            "quality_score": quality_score,
        }