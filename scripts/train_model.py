"""
模型训练脚本

用法:
    python -m scripts.train_model --model bilstm --epochs 100 --batch_size 32
    python -m scripts.train_model --model stgcn --input_mode keypoints
    python -m scripts.train_model --model transformer --input_mode features
"""

import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset import create_dataloaders
from src.models.model_factory import create_model, list_available_models
from src.models.trainer import Trainer, set_seed, get_device
from src.models.data_types import ModelConfig
from src.utils.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="训练动作分类模型")

    parser.add_argument(
        "--model", type=str, default=None,
        help=f"模型类型: {', '.join(list_available_models())}（默认从配置文件读取）",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=None, help="隐藏层维度")
    parser.add_argument(
        "--input_mode", type=str, default=None,
        choices=["keypoints", "features"], help="输入模式",
    )
    parser.add_argument("--templates_dir", type=str, default=None, help="模板目录")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="检查点保存目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--augment", type=int, default=None, help="每个模板增强样本数")

    return parser.parse_args()


def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train_model")

    # 加载配置
    config = get_config()
    model_cfg = config.get("model", {})
    actions_cfg = config.get("actions", {})
    aug_cfg = config.get("augmentation", {})

    # 解析命令行参数（命令行优先于配置文件）
    args = parse_args()

    model_type = args.model or model_cfg.get("backbone_type", "bilstm")
    epochs = args.epochs or model_cfg.get("epochs", 100)
    batch_size = args.batch_size or model_cfg.get("batch_size", 32)
    lr = args.lr or model_cfg.get("learning_rate", 1e-3)
    hidden_dim = args.hidden_dim or model_cfg.get("hidden_dim", 128)
    input_mode = args.input_mode or model_cfg.get("input_mode", "keypoints")
    templates_dir = args.templates_dir or actions_cfg.get("templates_dir", "data/templates")
    checkpoint_dir = args.checkpoint_dir or model_cfg.get("checkpoint_dir", "outputs/checkpoints")
    seed = args.seed or model_cfg.get("seed", 42)
    augment_per_template = args.augment or aug_cfg.get("augment_per_template", 5)

    # 固定随机种子
    set_seed(seed)
    device = get_device()

    logger.info("=" * 60)
    logger.info("动作分类模型训练")
    logger.info("=" * 60)
    logger.info(f"模型类型: {model_type}")
    logger.info(f"输入模式: {input_mode}")
    logger.info(f"训练轮数: {epochs}")
    logger.info(f"批大小: {batch_size}")
    logger.info(f"学习率: {lr}")
    logger.info(f"隐藏维度: {hidden_dim}")
    logger.info(f"模板目录: {templates_dir}")
    logger.info(f"检查点目录: {checkpoint_dir}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"增强倍数: {augment_per_template}")
    logger.info("=" * 60)

    # 构建数据集
    target_frames = model_cfg.get("target_frames", 60)
    train_loader, val_loader, test_loader = create_dataloaders(
        templates_dir=templates_dir,
        target_frames=target_frames,
        input_mode=input_mode,
        augment_per_template=augment_per_template,
        batch_size=batch_size,
        seed=seed,
    )

    # 从数据集推断参数
    dataset = train_loader.dataset
    # Subset 的 dataset 属性指向原始 ActionDataset
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    num_classes = base_dataset.num_classes
    input_dim = base_dataset.feature_dim if input_mode == "features" else model_cfg.get("input_dim", 20)

    logger.info(f"数据集: 训练={len(train_loader.dataset)}, "
                f"验证={len(val_loader.dataset)}, 测试={len(test_loader.dataset)}")
    logger.info(f"类别数: {num_classes}, 输入维度: {input_dim}")

    # 创建模型
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=model_cfg.get("dropout", 0.3),
        num_layers=model_cfg.get("num_layers", 2),
        num_heads=model_cfg.get("num_heads", 4),
        in_channels=model_cfg.get("in_channels", 4),
    )
    logger.info(f"模型参数量: {model.get_num_parameters():,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=model_cfg.get("weight_decay", 1e-4),
        lambda_quality=model_cfg.get("lambda_quality", 0.5),
        patience=model_cfg.get("patience", 10),
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # 开始训练
    history = trainer.train(num_epochs=epochs)

    # 绘制训练曲线
    curves_path = str(Path(checkpoint_dir) / f"{model_type}_training_curves.png")
    trainer.plot_training_curves(curves_path)

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳验证准确率: {trainer.best_val_acc:.4f}")
    logger.info(f"检查点: {checkpoint_dir}")
    logger.info(f"训练曲线: {curves_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()