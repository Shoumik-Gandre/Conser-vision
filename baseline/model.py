import torch
from torchvision import models


def get_baseline_model() -> torch.nn.Module:
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
    return model


def load_baseline_model(model_path: str, device: torch.device) -> torch.nn.Module:
    return torch.load(model_path, map_location=device)
