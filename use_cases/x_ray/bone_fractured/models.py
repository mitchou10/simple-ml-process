import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(3, 224, 224)):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # calcul auto de la taille
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            out = self._forward_features(dummy)
            n_features = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TransferLearningModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        model_name="resnet18",
        pretrained=True,
        freeze_features=True,
    ):
        super(TransferLearningModel, self).__init__()

        # Charger le modèle pré-entraîné
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Retirer la couche de classification
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Geler les paramètres du backbone si demandé
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Nouvelle tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def unfreeze_backbone(self):
        """Dégeler le backbone pour fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """Geler le backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False


def get_model(model_type="simple_cnn", **kwargs):
    if model_type == "simple_cnn":
        return SimpleCNN(**kwargs)
    elif model_type == "transfer_learning":
        return TransferLearningModel(**kwargs)
    else:
        raise ValueError(f"Model type {model_type} not recognized")
