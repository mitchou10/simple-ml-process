import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import os
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


class Trainer:

    def __init__(self, model: nn.Module, experiment_name: str = "Default"):
        logger.info(f"Initializing Trainer with experiment: {experiment_name}")
        self.model = model
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # Configuration MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Base de données locale

        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # L'expériment existe déjà
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        device: torch.device = None,
        log_interval: int = 10,
    ):
        """
        Fonction d'entraînement pour le modèle avec logging MLflow
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)

        # Critère et optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Démarrage du run MLflow
        with mlflow.start_run():

            # Log des hyperparamètres
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("model_type", type(self.model).__name__)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", train_loader.batch_size)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("criterion", "CrossEntropyLoss")
            mlflow.log_param("weight_decay", 1e-5)
            mlflow.log_param("device", str(device))

            # Log des informations du modèle
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            mlflow.log_param("total_parameters", total_params)
            mlflow.log_param("trainable_parameters", trainable_params)

            # Log des informations du dataset
            mlflow.log_param("train_size", len(train_loader.dataset))
            if val_loader:
                mlflow.log_param("val_size", len(val_loader.dataset))

            for epoch in range(epochs):
                # Phase d'entraînement
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info("-" * 50)

                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(device), targets.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Statistiques
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    # Log des métriques par batch (optionnel)
                    if batch_idx % log_interval == 0:
                        step = epoch * len(train_loader) + batch_idx
                        mlflow.log_metric("batch_loss", loss.item(), step=step)

                        logger.info(
                            f"Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}"
                        )

                # Métriques d'entraînement pour l'époque
                train_loss = running_loss / len(train_loader)
                train_acc = 100 * correct / total

                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)

                # Log des métriques d'époque
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)

                logger.info(
                    f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%"
                )

                # Phase de validation
                if val_loader is not None:
                    val_loss, val_acc = self.validate_model(
                        val_loader, criterion, device
                    )
                    self.history["val_loss"].append(val_loss)
                    self.history["val_acc"].append(val_acc)

                    # Log des métriques de validation
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                    logger.info(
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%"
                    )

            # Log du modèle final
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name=f"BoneFracturedModel_{mlflow.active_run().info.run_id[:8]}",
            )

            # Log des métriques finales
            final_train_acc = self.history["train_acc"][-1]
            mlflow.log_metric("final_train_accuracy", final_train_acc)

            if val_loader is not None:
                final_val_acc = self.history["val_acc"][-1]
                mlflow.log_metric("final_val_accuracy", final_val_acc)

                # Log de la différence pour détecter l'overfitting
                overfitting_gap = final_train_acc - final_val_acc
                mlflow.log_metric("overfitting_gap", overfitting_gap)

        return self.history

    def validate_model(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ):
        """
        Fonction de validation
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)

                outputs = self.model(data)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return val_loss / len(val_loader), 100 * correct / total

    def evaluate_model(self, test_loader: DataLoader, device: torch.device = None):
        """
        Fonction d'évaluation sur le test set avec logging MLflow
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Statistiques par classe
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] = (
                        class_correct.get(label, 0)
                        + (predicted[i] == targets[i]).item()
                    )
                    class_total[label] = class_total.get(label, 0) + 1

        accuracy = 100 * correct / total

        # Log des métriques de test si dans un run actif
        if mlflow.active_run():
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_total_samples", total)
            mlflow.log_metric("test_correct_predictions", correct)

            # Log des métriques par classe
            for class_id, class_correct_count in class_correct.items():
                class_accuracy = 100 * class_correct_count / class_total[class_id]
                mlflow.log_metric(f"test_accuracy_class_{class_id}", class_accuracy)

        return accuracy

    def log_confusion_matrix(
        self,
        test_loader: DataLoader,
        class_names: list = None,
        device: torch.device = None,
    ):
        """
        Log de la matrice de confusion
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calcul de la matrice de confusion
        cm = confusion_matrix(all_targets, all_predictions)

        # Création du graphique
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or [f"Class {i}" for i in range(len(cm))],
            yticklabels=class_names or [f"Class {i}" for i in range(len(cm))],
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        # Sauvegarde et log
        plt.tight_layout()
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)

        if mlflow.active_run():
            mlflow.log_artifact(confusion_matrix_path)

            # Log du rapport de classification
            report = classification_report(
                all_targets, all_predictions, target_names=class_names, output_dict=True
            )

            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)

        plt.close()

        # Nettoyage
        if os.path.exists(confusion_matrix_path):
            os.remove(confusion_matrix_path)
