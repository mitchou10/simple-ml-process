import sys
import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from use_cases.x_ray.bone_fractured.models import get_model
from use_cases.x_ray.bone_fractured.datasets.loader import (
    create_datasets,
    BoneFracturedDatasetLoader,
)
from loguru import logger
from train.process.main import Trainer

logger.remove()
logger.add(sys.stderr, level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train bone fracture detection model")

    # Model parameters
    parser.add_argument(
        "--model-type",
        default="transfer_learning",
        choices=["transfer_learning", "custom"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--model-name",
        default="resnet18",
        help="Name of the model (e.g., resnet18, resnet50)",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of workers for data loading"
    )

    # Data parameters
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation for training",
    )

    # Device
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configuration du logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Configuration du device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Crée les datasets
    loader = BoneFracturedDatasetLoader()
    train_dataset, val_dataset, test_dataset = create_datasets(
        loader, image_size=tuple(args.image_size), augment_train=not args.no_augment
    )

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    logger.info(f"Class distribution: {train_dataset.get_class_counts()}")

    # Crée les dataloaders avec équilibrage des classes
    train_weights = train_dataset.get_sample_weights()
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialise le modèle avec la bonne taille d'entrée
    model = get_model(
        model_type=args.model_type,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )

    # Affiche le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(model)
    history = trainer.train_model(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    # Évaluation finale sur le test set
    test_accuracy = trainer.evaluate_model(test_loader, device)
    trainer.log_confusion_matrix(
        test_loader, class_names=["No Fracture", "Fracture"], device=device
    )
