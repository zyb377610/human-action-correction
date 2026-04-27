"""
模型评估脚本

用法:
    python -m scripts.evaluate_model --checkpoint outputs/checkpoints/best_model.pth --model bilstm
    python -m scripts.evaluate_model --checkpoint outputs/checkpoints/best_model.pth --model stgcn --input_mode keypoints
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.models.dataset import create_dataloaders
from src.models.model_factory import create_model, list_available_models
from src.models.trainer import get_device, set_seed
from src.utils.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="评估动作分类模型")

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"模型类型: {', '.join(list_available_models())}",
    )
    parser.add_argument(
        "--input_mode", type=str, default=None,
        choices=["keypoints", "features"], help="输入模式",
    )
    parser.add_argument("--templates_dir", type=str, default=None, help="模板目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="评估结果输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list, save_path: str
):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="预测标签",
        ylabel="真实标签",
        title="混淆矩阵",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子中显示数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("evaluate_model")

    args = parse_args()

    # 加载配置
    config = get_config()
    model_cfg = config.get("model", {})
    actions_cfg = config.get("actions", {})

    model_type = args.model or model_cfg.get("backbone_type", "bilstm")
    input_mode = args.input_mode or model_cfg.get("input_mode", "keypoints")
    templates_dir = args.templates_dir or actions_cfg.get("templates_dir", "data/templates")

    set_seed(args.seed)
    device = get_device()

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("模型评估")
    logger.info("=" * 60)
    logger.info(f"检查点: {args.checkpoint}")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"输入模式: {input_mode}")

    # 构建数据集（仅使用测试集）
    target_frames = model_cfg.get("target_frames", 60)
    _, _, test_loader = create_dataloaders(
        templates_dir=templates_dir,
        target_frames=target_frames,
        input_mode=input_mode,
        augment_per_template=config.get("augmentation", {}).get("augment_per_template", 5),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 推断参数
    dataset = test_loader.dataset
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    num_classes = base_dataset.num_classes
    class_names = base_dataset.class_names
    input_dim = base_dataset.feature_dim if input_mode == "features" else model_cfg.get("input_dim", 20)

    logger.info(f"测试集大小: {len(test_loader.dataset)}")
    logger.info(f"类别: {class_names}")

    # 创建并加载模型
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.3),
        num_layers=model_cfg.get("num_layers", 2),
        num_heads=model_cfg.get("num_heads", 4),
        in_channels=model_cfg.get("in_channels", 4),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"模型参数量: {model.get_num_parameters():,}")
    logger.info(f"训练最佳 accuracy: {checkpoint.get('best_accuracy', 'N/A')}")

    # 推理
    all_preds = []
    all_labels = []
    all_quality_preds = []
    all_quality_targets = []

    with torch.no_grad():
        for batch in test_loader:
            sequences, labels, quality_scores, masks = batch
            sequences = sequences.to(device)
            masks = masks.to(device)

            output = model(sequences, masks)
            preds = output["cls_logits"].argmax(dim=1).cpu().numpy()
            quality_pred = output["quality_score"].squeeze(-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_quality_preds.extend(quality_pred)
            all_quality_targets.extend(quality_scores.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 分类报告
    logger.info("\n" + "=" * 60)
    logger.info("分类报告")
    logger.info("=" * 60)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    )
    logger.info("\n" + report)

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"总体准确率: {accuracy:.4f}")

    # 保存分类报告
    report_path = output_dir / f"{model_type}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"模型: {model_type}\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"总体准确率: {accuracy:.4f}\n\n")
        f.write(report)
    logger.info(f"分类报告已保存: {report_path}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = str(output_dir / f"{model_type}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    logger.info(f"混淆矩阵已保存: {cm_path}")

    # 质量评分 MAE
    quality_mae = np.mean(np.abs(
        np.array(all_quality_preds) - np.array(all_quality_targets)
    ))
    logger.info(f"质量评分 MAE: {quality_mae:.2f}")

    logger.info("=" * 60)
    logger.info("评估完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()