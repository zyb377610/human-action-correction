"""
模型训练流水线

包含 Trainer（训练循环）、MultiTaskLoss（多任务损失）、
EarlyStopping（早停策略）和训练工具函数。
"""

import copy
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

from .base_model import BaseActionModel

logger = logging.getLogger(__name__)


# ===== 工具函数 =====

def set_seed(seed: int = 42):
    """
    固定所有随机种子，保证训练可复现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置: {seed}")


def get_device() -> torch.device:
    """自动检测并返回可用设备 (CUDA / CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("使用 CPU")
    return device


# ===== 多任务损失 =====

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数

    L_total = L_cls + lambda_quality * L_quality
    - L_cls: CrossEntropyLoss（分类损失）
    - L_quality: MSELoss（质量评分回归损失）
    """

    def __init__(self, lambda_quality: float = 0.5):
        super().__init__()
        self.lambda_quality = lambda_quality
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.quality_loss_fn = nn.MSELoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        quality_pred: torch.Tensor,
        labels: torch.Tensor,
        quality_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_logits: (B, num_classes) 分类 logits
            quality_pred: (B, 1) 质量评分预测
            labels: (B,) 类别标签
            quality_target: (B,) 质量评分真值

        Returns:
            dict 含 "total", "cls", "quality" 三个损失值
        """
        loss_cls = self.cls_loss_fn(cls_logits, labels)
        loss_quality = self.quality_loss_fn(
            quality_pred.squeeze(-1), quality_target
        )
        loss_total = loss_cls + self.lambda_quality * loss_quality

        return {
            "total": loss_total,
            "cls": loss_cls,
            "quality": loss_quality,
        }


# ===== 早停策略 =====

class EarlyStopping:
    """
    早停策略

    当监控指标连续 patience 个 epoch 不提升时触发早停。
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience: 耐心值
            min_delta: 最小改善阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        更新早停状态

        Args:
            score: 当前 epoch 的监控指标（越大越好）

        Returns:
            True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"早停触发: 连续 {self.patience} 个 epoch 无改善 "
                    f"(最佳: {self.best_score:.4f})"
                )

        return self.early_stop


# ===== 训练器 =====

class Trainer:
    """
    模型训练器

    提供完整的训练循环：训练 → 验证 → 早停 → 检查点 → 日志 → 曲线。

    使用示例:
        trainer = Trainer(model, train_loader, val_loader)
        trainer.train(num_epochs=100)
        trainer.plot_training_curves("outputs/curves.png")
    """

    def __init__(
        self,
        model: BaseActionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_quality: float = 0.5,
        patience: int = 10,
        checkpoint_dir: str = "outputs/checkpoints",
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化器和调度器
        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = None  # 在 train() 中初始化

        # 损失函数和早停
        self.criterion = MultiTaskLoss(lambda_quality=lambda_quality)
        self.early_stopping = EarlyStopping(patience=patience)

        # 检查点目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 训练历史
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.best_model_state = None

    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch

        Returns:
            (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            sequences, labels, quality_scores, masks = batch
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            quality_scores = quality_scores.float().to(self.device)
            masks = masks.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(sequences, masks)

            # 计算损失
            losses = self.criterion(
                output["cls_logits"], output["quality_score"],
                labels, quality_scores,
            )

            # 反向传播
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += losses["total"].item() * labels.size(0)
            preds = output["cls_logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, float]:
        """
        验证一个 epoch

        Returns:
            (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            sequences, labels, quality_scores, masks = batch
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            quality_scores = quality_scores.float().to(self.device)
            masks = masks.to(self.device)

            output = self.model(sequences, masks)
            losses = self.criterion(
                output["cls_logits"], output["quality_score"],
                labels, quality_scores,
            )

            total_loss += losses["total"].item() * labels.size(0)
            preds = output["cls_logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        完整训练循环

        Args:
            num_epochs: 最大训练轮数

        Returns:
            训练历史字典
        """
        logger.info(
            f"开始训练: {num_epochs} epochs, "
            f"模型参数量: {self.model.get_num_parameters():,}"
        )

        # 初始化余弦退火调度器
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate_epoch()

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint(
                    self.checkpoint_dir / "best_model.pth",
                    epoch, val_acc,
                )
                logger.info(f"  ★ 新最佳模型: val_acc={val_acc:.4f}")

            # 早停检查
            if self.early_stopping(val_acc):
                logger.info(f"早停于 epoch {epoch}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"训练完成，加载最佳模型 (val_acc={self.best_val_acc:.4f})")

        # 保存最终检查点
        self.save_checkpoint(
            self.checkpoint_dir / "last_model.pth",
            len(self.history["train_loss"]),
            self.best_val_acc,
        )

        return self.history

    def save_checkpoint(self, path: str, epoch: int, best_acc: float):
        """保存训练检查点"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": best_acc,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, str(path))
        logger.debug(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载训练检查点，恢复训练状态"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"检查点不存在: {path}")

        checkpoint = torch.load(str(path), map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint.get("best_accuracy", 0.0)
        self.history = checkpoint.get("history", self.history)

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"检查点已加载: epoch={checkpoint['epoch']}, "
            f"best_acc={self.best_val_acc:.4f}"
        )

    def plot_training_curves(self, save_path: str = "outputs/training_curves.png"):
        """
        绘制训练曲线图（loss + accuracy）

        Args:
            save_path: 图片保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss 曲线
        axes[0].plot(epochs, self.history["train_loss"], "b-", label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"], "r-", label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy 曲线
        axes[1].plot(epochs, self.history["train_acc"], "b-", label="Train Acc")
        axes[1].plot(epochs, self.history["val_acc"], "r-", label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"训练曲线已保存: {save_path}")