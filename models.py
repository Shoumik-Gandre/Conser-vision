import typing
from typing import Tuple

import torch
import torchvision

from enumerations import Architectures


class ModelWrapper(typing.Protocol):

    def classifier(self) -> torch.nn.Module:
        ...

    def transforms(self) -> torch.nn.Module:
        ...

    @property
    def model(self) -> torch.nn.Module:
        return ...


class ResNet50Wrapper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(2048, 8)
        self.model = model

    @property
    def classifier(self):
        return self.model.fc

    @property
    def transforms(self):
        return torchvision.models.ResNet50_Weights.DEFAULT.transforms()


class ResNet152Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
        model.fc = torch.nn.Linear(2048, 8)
        self.model = model

    @property
    def classifier(self):
        return self.model.fc

    @property
    def transforms(self):
        return torchvision.models.ResNet152_Weights.DEFAULT.transforms()


class EfficientNetV2LWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(1280, 8)
        self.model = model

    @property
    def classifier(self):
        return self.model.classifier[1]

    @property
    def transforms(self):
        return torchvision.models.EfficientNet_V2_L_Weights.DEFAULT.transforms()


def get_model(architecture: Architectures) -> ModelWrapper:
    """returns Model, Transforms"""
    match architecture:
        case Architectures.RESNET50:
            return ResNet50Wrapper()

        case Architectures.RESNET152:
            return ResNet152Wrapper()

        case Architectures.EFFICIENT_NET:
            return EfficientNetV2LWrapper()


def load_model(
        model_path: str,
        architecture: Architectures,
        device: torch.device
) -> Tuple[torch.nn.Module, torchvision.transforms.Compose]:
    """returns Model, Transforms"""
    transforms = None
    match architecture:
        case Architectures.RESNET50:
            transforms = torchvision.models.ResNet50_Weights.DEFAULT.transforms()

        case Architectures.RESNET152:
            transforms = torchvision.models.ResNet152_Weights.DEFAULT.transforms()

        case Architectures.EFFICIENT_NET:
            transforms = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT.transforms()

    return torch.load(model_path, map_location=device), transforms
