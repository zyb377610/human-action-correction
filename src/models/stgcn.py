"""
ST-GCN 时空图卷积网络

基于人体骨骼拓扑图的时空图卷积，直接处理关键点序列 (B, T, V, C)。
参考: Yan et al. "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition" (AAAI 2018)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseActionModel
from .skeleton_graph import get_normalized_adjacency


class SpatialGraphConv(nn.Module):
    """
    空间图卷积层

    对每一帧执行图卷积: X' = A_norm @ X @ W
    """

    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)  # (V, V) 归一化邻接矩阵
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            (B, C_out, T, V)
        """
        # 图卷积: 沿节点维度聚合邻居特征
        # x: (B, C, T, V) @ adj: (V, V) -> (B, C, T, V)
        x = torch.einsum("bctv,vw->bctw", x, self.adj)
        # 1x1 卷积变换通道
        x = self.conv(x)
        x = self.bn(x)
        return x


class STGCNBlock(nn.Module):
    """
    ST-GCN 基本块: 空间图卷积 + 时间卷积 + 残差连接
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels, adj)

        # 时间卷积: kernel_size=9 沿时间维度
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        # 残差连接
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            (B, C_out, T', V)
        """
        res = self.residual(x)

        x = self.spatial_conv(x)
        x = self.relu(x)
        x = self.temporal_conv(x)
        x = self.dropout(x)

        x = x + res
        x = self.relu(x)
        return x


class STGCN(BaseActionModel):
    """
    ST-GCN 时空图卷积网络

    架构: 3 层 STGCNBlock + 全局平均池化 + 双头输出
    输入: (B, T, V, C) 其中 V=33, C=4 (x, y, z, visibility)
    输出: cls_logits (B, num_classes), quality_score (B, 1)
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        # 归一化邻接矩阵
        adj_np = get_normalized_adjacency()
        adj = torch.from_numpy(adj_np)

        # 输入 batch norm
        self.input_bn = nn.BatchNorm1d(in_channels * 33)

        # ST-GCN 层
        channels = [in_channels] + [hidden_dim * (2 ** min(i, 2)) for i in range(num_layers)]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                STGCNBlock(channels[i], channels[i + 1], adj,
                           stride=1, dropout=dropout)
            )

        final_dim = channels[-1]

        # 双头输出
        self.cls_head = nn.Linear(final_dim, num_classes)
        self.quality_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, V, C) 关键点序列，V=33, C=4
            mask: (B, T) padding mask（未使用，ST-GCN 通过全局池化自然处理）

        Returns:
            {"cls_logits": (B, num_classes), "quality_score": (B, 1)}
        """
        B, T, V, C = x.shape

        # 输入归一化: (B, T, V, C) -> (B, V*C) per frame -> BN -> reshape
        x_flat = x.reshape(B, T, V * C)  # (B, T, V*C)
        x_flat = x_flat.permute(0, 2, 1)  # (B, V*C, T)
        x_flat = self.input_bn(x_flat)
        x_flat = x_flat.permute(0, 2, 1)  # (B, T, V*C)
        x = x_flat.reshape(B, T, V, C)

        # 转为 (B, C, T, V) 适配卷积
        x = x.permute(0, 3, 1, 2)  # (B, C, T, V)

        # ST-GCN 层
        for layer in self.layers:
            x = layer(x)

        # 全局平均池化: (B, C_final, T, V) -> (B, C_final)
        x = x.mean(dim=[2, 3])  # 时间 + 节点维度平均

        # 双头输出
        cls_logits = self.cls_head(x)
        quality_score = self.quality_head(x) * 100.0  # [0, 100]

        return {
            "cls_logits": cls_logits,
            "quality_score": quality_score,
        }