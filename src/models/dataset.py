"""
动作分类数据集

基于 TemplateLibrary 和数据增强构建 PyTorch Dataset，
支持关键点和特征向量两种输入模式，提供 train/val/test 划分。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.pose_estimation.data_types import PoseSequence
from src.pose_estimation.feature_extractor import FeatureExtractor
from src.data.template_library import TemplateLibrary
from src.data.augmentation import augment_batch

logger = logging.getLogger(__name__)


class ActionDataset(Dataset):
    """
    动作分类 PyTorch 数据集

    从模板库加载标准动作，通过数据增强扩充样本，
    支持 keypoints (T, 33, 4) 和 features (T, D) 两种输入模式。

    使用示例:
        dataset = ActionDataset(templates_dir="data/templates", target_frames=60)
        x, label, quality, mask = dataset[0]
    """

    def __init__(
        self,
        templates_dir: Optional[str] = None,
        target_frames: int = 60,
        input_mode: str = "keypoints",
        augment_per_template: int = 5,
        time_warp_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.005,
        mirror_enabled: bool = True,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            templates_dir: 模板根目录路径
            target_frames: 序列统一长度
            input_mode: 输入模式 ("keypoints" 或 "features")
            augment_per_template: 每个模板生成的增强样本数
            time_warp_range: 时间拉伸范围
            noise_std: 噪声标准差
            mirror_enabled: 是否启用镜像增强
            seed: 随机种子
        """
        super().__init__()
        self.target_frames = target_frames
        self.input_mode = input_mode
        self._feature_extractor = FeatureExtractor() if input_mode == "features" else None

        # 加载模板库
        self._library = TemplateLibrary(templates_dir)
        actions = self._library.list_actions()

        if not actions:
            logger.warning("模板库为空，数据集将无样本")

        # 构建类别映射 (按字母序排列保证一致性)
        self.class_names: List[str] = sorted(actions)
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        # 加载并增强数据
        self._samples: List[Tuple[np.ndarray, int, float]] = []
        self._build_samples(augment_per_template, time_warp_range,
                            noise_std, mirror_enabled, seed)

        logger.info(
            f"ActionDataset: {len(self._samples)} 样本, "
            f"{len(self.class_names)} 类别, "
            f"input_mode={input_mode}, target_frames={target_frames}"
        )

    def _build_samples(
        self,
        augment_per_template: int,
        time_warp_range: Tuple[float, float],
        noise_std: float,
        mirror_enabled: bool,
        seed: Optional[int],
    ):
        """从模板库构建所有样本（原始 + 增强）"""
        for action_name in self.class_names:
            label_idx = self.class_to_idx[action_name]
            templates = self._library.load_all_templates(action_name)

            for tmpl_name, sequence in templates.items():
                # 原始模板（质量评分 = 100）
                arr = self._convert_sequence(sequence)
                self._samples.append((arr, label_idx, 100.0))

                # 增强样本（质量评分 = 70~95，模拟非标准变体）
                if augment_per_template > 0:
                    aug_sequences = augment_batch(
                        sequence,
                        num_augmented=augment_per_template,
                        time_warp_range=time_warp_range,
                        noise_std=noise_std,
                        mirror_enabled=mirror_enabled,
                        seed=seed,
                    )
                    rng = np.random.default_rng(seed)
                    for aug_seq in aug_sequences:
                        aug_arr = self._convert_sequence(aug_seq)
                        # 增强样本质量稍低
                        quality = float(rng.uniform(70.0, 95.0))
                        self._samples.append((aug_arr, label_idx, quality))

    def _convert_sequence(self, sequence: PoseSequence) -> np.ndarray:
        """
        将 PoseSequence 转为 numpy 数组

        Returns:
            keypoints 模式: (T, 33, 4)
            features 模式: (T, D)
        """
        if self.input_mode == "features":
            return self._feature_extractor.extract_sequence_features(sequence)
        else:
            return sequence.to_numpy()

    def _pad_or_truncate(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将序列 pad 或 truncate 至 target_frames

        Args:
            arr: 原始序列，shape (T, ...) 

        Returns:
            (padded_arr, mask): padded_arr shape (target_frames, ...),
                                mask shape (target_frames,)，1=有效帧，0=填充帧
        """
        T = arr.shape[0]
        rest_shape = arr.shape[1:]

        if T >= self.target_frames:
            # 截断
            padded = arr[:self.target_frames]
            mask = np.ones(self.target_frames, dtype=np.float32)
        else:
            # 填充
            padded = np.zeros((self.target_frames, *rest_shape), dtype=np.float32)
            padded[:T] = arr
            mask = np.zeros(self.target_frames, dtype=np.float32)
            mask[:T] = 1.0

        return padded.astype(np.float32), mask

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, torch.Tensor]:
        """
        Returns:
            (sequence_tensor, label, quality_score, padding_mask)
            - sequence_tensor: (target_frames, 33, 4) 或 (target_frames, D)
            - label: 类别索引 int
            - quality_score: 质量评分 float [0, 100]
            - padding_mask: (target_frames,) 1=有效，0=填充
        """
        arr, label, quality = self._samples[idx]
        padded, mask = self._pad_or_truncate(arr)

        return (
            torch.from_numpy(padded),
            label,
            quality,
            torch.from_numpy(mask),
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def feature_dim(self) -> int:
        """特征维度（features 模式下每帧的特征数）"""
        if self.input_mode == "features":
            return self._feature_extractor.feature_dim
        else:
            return 33 * 4  # keypoints 展平


def split_dataset(
    dataset: ActionDataset,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    按比例划分数据集为 train/val/test

    Args:
        dataset: 完整数据集
        ratios: (train, val, test) 比例，和为 1
        seed: 随机种子

    Returns:
        (train_subset, val_subset, test_subset)
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"比例之和必须为 1，当前为 {sum(ratios)}"

    n = len(dataset)
    indices = list(range(n))

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def create_dataloaders(
    templates_dir: Optional[str] = None,
    target_frames: int = 60,
    input_mode: str = "keypoints",
    augment_per_template: int = 5,
    batch_size: int = 32,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    工厂函数：创建训练/验证/测试 DataLoader

    Args:
        templates_dir: 模板根目录
        target_frames: 序列统一长度
        input_mode: 输入模式
        augment_per_template: 每个模板增强样本数
        batch_size: 批大小
        split_ratios: 划分比例
        seed: 随机种子
        num_workers: DataLoader 工作线程数

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = ActionDataset(
        templates_dir=templates_dir,
        target_frames=target_frames,
        input_mode=input_mode,
        augment_per_template=augment_per_template,
        seed=seed,
    )

    train_set, val_set, test_set = split_dataset(dataset, split_ratios, seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    logger.info(
        f"DataLoaders 创建完成: train={len(train_set)}, "
        f"val={len(val_set)}, test={len(test_set)}, batch_size={batch_size}"
    )

    return train_loader, val_loader, test_loader