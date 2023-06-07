from typing import Tuple

import torch
import torchvision
from torchvision import models

from enumerations import Architectures


def _get_resnet50_model() -> Tuple[models.ResNet, torchvision.transforms.Compose]:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(  # type: ignore
        torch.nn.Linear(2048, 1024),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(1024, 512),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(512, 256),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(256, 128),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(128, 8),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    return model, models.ResNet50_Weights.DEFAULT.transforms()


def _get_resnet152_model() -> Tuple[models.ResNet, torchvision.transforms.Compose]:
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(  # type: ignore
        torch.nn.Linear(2048, 1024),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(1024, 512),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(512, 256),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(256, 128),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.33),  # common technique to mitigate overfitting
        torch.nn.Linear(128, 8),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    return model, models.ResNet152_Weights.DEFAULT.transforms()


def _get_efficient_net_model() -> Tuple[models.EfficientNet, torchvision.transforms.Compose]:
    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(1280, 8)
    return model, models.ResNet50_Weights.DEFAULT.transforms()


def get_model(architecture: Architectures) -> Tuple[torch.nn.Module, torchvision.transforms.Compose]:
    """returns Model, Transforms"""
    match architecture:
        case Architectures.RESNET50:
            return _get_resnet50_model()

        case Architectures.RESNET152:
            return _get_resnet152_model()

        case Architectures.EFFICIENT_NET:
            return _get_efficient_net_model()


def load_model(
        model_path: str,
        architecture: Architectures,
        device: torch.device
) -> Tuple[torch.nn.Module, torchvision.transforms.Compose]:
    """returns Model, Transforms"""
    transforms = None
    match architecture:
        case Architectures.RESNET50:
            transforms = models.ResNet50_Weights.DEFAULT.transforms()

        case Architectures.RESNET152:
            transforms = models.ResNet152_Weights.DEFAULT.transforms()

        case Architectures.EFFICIENT_NET:
            transforms = models.EfficientNet_V2_L_Weights.DEFAULT.transforms()

    return torch.load(model_path, map_location=device), transforms
