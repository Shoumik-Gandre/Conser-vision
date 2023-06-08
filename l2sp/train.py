import pathlib

import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split
from dataset import ImagesDataset
from enumerations import Architectures
from l2sp.trainer import TrainerArgs, L2SPTrainer, LSquareStartingPointRegularization
import torch

from models import get_model


def train_l2sp(
        features_csv: str,
        labels_csv: str,
        images_dir: str,
        model_dir: str,
        epochs: int,
        batch_size: int,
) -> None:
    train_features = pd.read_csv(features_csv, index_col="id")
    train_features['filepath'] = train_features['filepath'].apply(
        lambda path: pathlib.Path(images_dir) / str(path))
    train_labels = pd.read_csv(labels_csv, index_col="id")

    y = train_labels
    x = train_features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImagesDataset(x_train, y_train)
    eval_dataset = ImagesDataset(x_eval, y_eval)
    training_args = TrainerArgs(epochs=epochs, batch_size=batch_size, model_dir=model_dir)
    model = get_l2sp_model()
    pretrained_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

    def requires_grad_false(param: torch.nn.Parameter) -> torch.nn.Parameter:
        param.requires_grad = False
        return param

    starting_params = {
        name: requires_grad_false(param)
        for name, param in pretrained_model.named_parameters()
        if (
                name.split('.')[0] != 'fc'  # It should not be a classifier
                and name.split('.')[-1] != 'bias'  # It should not be a bias
                and param.requires_grad  # It should be updatable
        )
    }

    sp_regularizer = LSquareStartingPointRegularization(
        starting_parameters=starting_params,
        coefficient=1e-2,
        device=device
    )

    trainer = L2SPTrainer(
        model=model,
        optimizer=torch.optim.SGD(
            [
                {
                    'params': [param for name, param in model.named_parameters() if name.split('.')[0] != 'fc'],
                },
                {
                    'params': [param for name, param in model.named_parameters() if name.split('.')[0] == 'fc'],
                    'weight_decay': 1e-3,
                },
            ],
            lr=1e-2,
            momentum=0.9
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        sp_regularizer=sp_regularizer,
        device=device,
    )
    trainer.train(train_args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)


def l2sp_train(
        features_csv: str,
        labels_csv: str,
        images_dir: str,
        model_path: str,
        model_arch: Architectures,
        hyperparameters
) -> None:
    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))
    train_labels = pd.read_csv(labels_csv, index_col="id")

    y = train_labels
    x = features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_wrapper = get_model(model_arch)
    pretrained_model = get_model(model_arch)
    train_dataset = ImagesDataset(x_train, y_train, model_wrapper.transforms())
    eval_dataset = ImagesDataset(x_eval, y_eval, model_wrapper.transforms())

    def requires_grad_false(param: torch.nn.Parameter) -> torch.nn.Parameter:
        param.requires_grad = False
        return param

    starting_params = {
        name: requires_grad_false(param)
        for name, param in pretrained_model.model.named_parameters()
        if (
                param != pretrained_model.classifier()
                and name.split('.')[-1] != 'bias'  # It should not be a bias
                and param.requires_grad  # It should be updatable
        )
    }

    sp_regularizer = LSquareStartingPointRegularization(
        starting_parameters=starting_params,
        coefficient=1e-2,
        device=device
    )

    trainer = L2SPTrainer(
        model=model_wrapper.model,
        optimizer=torch.optim.SGD(
            [
                {
                    'params': [
                        param for name, param in model_wrapper.model.named_parameters()
                        if param != model_wrapper.classifier()
                    ],
                },
                {
                    'params': [
                        param for name, param in model_wrapper.model.named_parameters()
                        if param == model_wrapper.classifier()
                    ],
                    'weight_decay': 1e-3,
                },
            ],
            lr=1e-2,
            momentum=0.9
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        sp_regularizer=sp_regularizer,
        device=device,
    )
    trainer.train(train_args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)