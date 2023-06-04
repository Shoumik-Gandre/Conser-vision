import torch
from torchvision import models


def get_baseline_model() -> torch.nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(  # type: ignore
        torch.nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
        torch.nn.BatchNorm1d(100),
        torch.nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        torch.nn.Dropout(0.1),  # common technique to mitigate overfitting
        torch.nn.Linear(100, 8),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    return model


def load_baseline_model(model_path: str, device: torch.device) -> torch.nn.Module:
    return torch.load(model_path, map_location=device)
