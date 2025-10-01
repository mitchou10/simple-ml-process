from glob import glob
import os
from train.dataset.loader import KaggleLoader
import re
import sys
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")


class BoneFracturedDatasetLoader(KaggleLoader):
    def __init__(self, download_path: str = os.path.join("data", "raw")):
        super().__init__(
            dataset_name="foyez767/x-ray-images-of-fractured-and-healthy-bones",
            download_path=download_path,
        )

    def load_data(self, *args, **kwargs) -> list[dict[str, str]]:
        # Implement data loading logic here
        counter = {"/Fractured/": 0, "/Non-Fractured/": 0}
        data = []
        for file in glob(os.path.join(self.download_path, "**/*.png"), recursive=True):
            # data/raw/X-ray Imaging Dataset for Detecting Fractured vs. Non-Fractured Bones/Original Dataset/Fractured/Fractured (121).png

            match = re.search(r"(Original Dataset|Augmented Dataset)", file)
            if match:
                sub_dataset_name = match.group(0)

                match = re.search(r"(/Fractured/|/Non-Fractured/)", file)
                if match:
                    label = match.group(0)
                    data.append(
                        {
                            "file_path": file,
                            "sub_dataset": sub_dataset_name,
                            "label": label,
                        }
                    )
                    counter[label] += 1
            else:
                raise RuntimeError(f"Could not determine sub-dataset for file: {file}")

        random.shuffle(data)
        return data

    def get_item(self, index: int | str, *args, **kwargs) -> dict[str, str]:
        data = self.load_data()
        if isinstance(index, int):
            return data[index]
        elif isinstance(index, str):
            for item in data:
                if item["file_path"] == index:
                    return item
        raise IndexError("Index out of range or file path not found.")

    def get_batch(
        self, indices: list[int | str], *args, **kwargs
    ) -> list[dict[str, str]]:
        return [self.get_item(index) for index in indices]


class BoneFracturedDataset(Dataset):
    def __init__(
        self, data: list[dict[str, str]], transform=None, target_transform=None
    ):
        """
        Dataset compatible avec torchvision

        Args:
            data: Liste des dictionnaires contenant les métadonnées des images
            transform: Transformations à appliquer aux images (torchvision.transforms)
            target_transform: Transformations à appliquer aux labels
        """
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

        # Mapping des labels vers des indices numériques
        self.label_to_idx = {"/Non-Fractured/": 0, "/Fractured/": 1}
        self.idx_to_label = {0: "/Non-Fractured/", 1: "/Fractured/"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retourne un tuple (image, label) compatible avec PyTorch/torchvision
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]

        # Chargement de l'image
        image_path = item["file_path"]
        try:
            image = Image.open(image_path).convert(
                "RGB"
            )  # Conversion en RGB pour compatibilité
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors du chargement de l'image {image_path}: {e}"
            )

        # Conversion du label en index numérique
        label_str = item["label"]
        label = self.label_to_idx[label_str]

        # Application des transformations
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_counts(self):
        """Retourne le nombre d'échantillons par classe"""
        counts = {"/Fractured/": 0, "/Non-Fractured/": 0}
        for item in self.data:
            counts[item["label"]] += 1
        return counts

    def get_sample_weights(self):
        """Calcule les poids pour équilibrer les classes"""
        class_counts = self.get_class_counts()
        total_samples = len(self.data)

        weights = []
        for item in self.data:
            label = item["label"]
            class_count = class_counts[label]
            weight = total_samples / (len(class_counts) * class_count)
            weights.append(weight)

        return torch.DoubleTensor(weights)


def get_default_transforms(image_size=(224, 224), augment=True):
    """
    Retourne les transformations par défaut pour le dataset

    Args:
        image_size: Taille cible des images (height, width)
        augment: Si True, applique des augmentations de données

    Returns:
        transform_train, transform_val: Transformations pour l'entraînement et la validation
    """

    # Transformations pour l'entraînement (avec augmentation)
    if augment:
        transform_train = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet stats
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Transformations pour la validation (sans augmentation)
    transform_val = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform_train, transform_val


def create_datasets(
    loader: BoneFracturedDatasetLoader,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    image_size: tuple = (224, 224),
    augment_train: bool = True,
    use_original_only: bool = True,
):
    """
    Crée les datasets d'entraînement, validation et test

    Args:
        loader: Instance de BoneFracturedDatasetLoader
        train_ratio: Proportion des données pour l'entraînement
        val_ratio: Proportion des données pour la validation
        test_ratio: Proportion des données pour le test
        image_size: Taille des images
        augment_train: Appliquer l'augmentation pour l'entraînement
        use_original_only: Utiliser seulement le dataset original (sans augmentation Kaggle)

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Chargement des données
    all_data = loader.load_data()

    # Filtrer pour utiliser seulement le dataset original si demandé
    if use_original_only:
        all_data = [
            item for item in all_data if item["sub_dataset"] == "Original Dataset"
        ]

    # Mélange et division des données
    import random

    random.shuffle(all_data)

    total_size = len(all_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size :]

    # Création des transformations
    transform_train, transform_val = get_default_transforms(image_size, augment_train)

    # Création des datasets
    train_dataset = BoneFracturedDataset(train_data, transform=transform_train)
    val_dataset = BoneFracturedDataset(val_data, transform=transform_val)
    test_dataset = BoneFracturedDataset(test_data, transform=transform_val)

    return train_dataset, val_dataset, test_dataset
