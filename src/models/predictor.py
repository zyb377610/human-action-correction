"""
模型推理接口

封装模型加载和推理逻辑，提供统一的 predict() API。
输入 PoseSequence，输出 PredictionResult。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.pose_estimation.data_types import PoseSequence
from src.pose_estimation.feature_extractor import FeatureExtractor

from .data_types import PredictionResult, ModelConfig
from .model_factory import create_model
from .trainer import get_device

logger = logging.getLogger(__name__)


class ActionPredictor:
    """
    动作分类推理器

    加载训练好的模型权重，对 PoseSequence 进行分类和质量评分。

    使用示例:
        predictor = ActionPredictor(
            checkpoint_path="outputs/checkpoints/best_model.pth",
            model_type="bilstm",
            class_names=["squat", "arm_raise", "side_bend", "lunge", "standing_stretch"],
        )
        result = predictor.predict(pose_sequence)
        print(result.label, result.confidence, result.quality_score)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "bilstm",
        class_names: Optional[List[str]] = None,
        config: Optional[ModelConfig] = None,
        input_mode: str = "keypoints",
        target_frames: int = 60,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            model_type: 模型类型
            class_names: 类别名称列表（按索引排序）
            config: 模型配置
            input_mode: 输入模式 ("keypoints" / "features")
            target_frames: 序列统一长度
            device: 推理设备（None 则自动检测）
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"检查点文件不存在: {self.checkpoint_path}"
            )

        self.device = device or get_device()
        self.input_mode = input_mode
        self.target_frames = target_frames
        self.class_names = class_names or []

        # 特征提取器（features 模式使用）
        self._feature_extractor = (
            FeatureExtractor() if input_mode == "features" else None
        )

        # 创建并加载模型
        if config is not None:
            self.model = create_model(model_type, config=config)
        else:
            num_classes = len(self.class_names) if self.class_names else 5
            input_dim = (
                self._feature_extractor.feature_dim
                if self._feature_extractor
                else 20
            )
            self.model = create_model(
                model_type,
                num_classes=num_classes,
                input_dim=input_dim,
            )

        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"ActionPredictor 初始化完成: model={model_type}, "
            f"device={self.device}, classes={len(self.class_names)}"
        )

    def _load_weights(self):
        """从 checkpoint 加载模型权重"""
        checkpoint = torch.load(
            str(self.checkpoint_path), map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"模型权重已加载: {self.checkpoint_path}")

    def _preprocess(self, sequence: PoseSequence) -> torch.Tensor:
        """
        预处理 PoseSequence -> 模型输入张量

        Returns:
            (1, target_frames, ...) 张量
        """
        # 转为 numpy
        if self.input_mode == "features":
            arr = self._feature_extractor.extract_sequence_features(sequence)
        else:
            arr = sequence.to_numpy()  # (T, 33, 4)

        # pad / truncate
        T = arr.shape[0]
        rest_shape = arr.shape[1:]

        if T >= self.target_frames:
            padded = arr[: self.target_frames]
        else:
            padded = np.zeros(
                (self.target_frames, *rest_shape), dtype=np.float32
            )
            padded[:T] = arr

        # 转为张量并添加 batch 维度
        tensor = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, sequence: PoseSequence) -> PredictionResult:
        """
        对单个 PoseSequence 进行推理

        Args:
            sequence: 姿态序列

        Returns:
            PredictionResult
        """
        x = self._preprocess(sequence)
        output = self.model(x)

        # 解析输出
        logits = output["cls_logits"]  # (1, num_classes)
        quality = output["quality_score"]  # (1, 1)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        quality_score = float(quality.squeeze().cpu())

        # 类别名称
        if self.class_names and pred_idx < len(self.class_names):
            label = self.class_names[pred_idx]
        else:
            label = str(pred_idx)

        # 各类概率字典
        class_probs = {}
        for i, prob in enumerate(probs):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            class_probs[name] = float(prob)

        return PredictionResult(
            label=label,
            confidence=confidence,
            quality_score=quality_score,
            class_probs=class_probs,
        )

    @torch.no_grad()
    def predict_batch(
        self, sequences: List[PoseSequence]
    ) -> List[PredictionResult]:
        """
        批量推理

        Args:
            sequences: PoseSequence 列表

        Returns:
            PredictionResult 列表
        """
        if not sequences:
            return []

        # 逐个预处理并拼接
        tensors = [self._preprocess(seq) for seq in sequences]
        batch = torch.cat(tensors, dim=0)  # (N, T, ...)

        output = self.model(batch)
        logits = output["cls_logits"]  # (N, num_classes)
        qualities = output["quality_score"]  # (N, 1)

        probs_batch = F.softmax(logits, dim=-1).cpu().numpy()
        qualities_np = qualities.squeeze(-1).cpu().numpy()

        results = []
        for i in range(len(sequences)):
            probs = probs_batch[i]
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            quality_score = float(qualities_np[i])

            label = (
                self.class_names[pred_idx]
                if pred_idx < len(self.class_names)
                else str(pred_idx)
            )

            class_probs = {}
            for j, prob in enumerate(probs):
                name = (
                    self.class_names[j] if j < len(self.class_names) else str(j)
                )
                class_probs[name] = float(prob)

            results.append(
                PredictionResult(
                    label=label,
                    confidence=confidence,
                    quality_score=quality_score,
                    class_probs=class_probs,
                )
            )

        return results