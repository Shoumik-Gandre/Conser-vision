import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import ImagesDataset
from enumerations import Architectures
from l2sp.hyperparams import L2SPHyperparams
from l2sp.trainer import LSquareStartingPointRegularization
import torch

from models import get_model
from l2sp.trainer import train


def l2sp_train(
        features_csv: str,
        labels_csv: str,
        images_dir: str,
        model_path: str,
        model_arch: Architectures,
        hyperparams: L2SPHyperparams,
) -> None:
    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))
    train_labels = pd.read_csv(labels_csv, index_col="id")

    y = train_labels
    x = features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_wrapper = get_model(model_arch)
    model_wrapper.model.to(device)
    transforms = model_wrapper.transforms
    pretrained_model = get_model(model_arch)
    pretrained_model.model.to(device)
    train_dataset = ImagesDataset(x_train, y_train, transforms)
    eval_dataset = ImagesDataset(x_eval, y_eval, transforms)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device=device_str
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=hyperparams.batch_size,
        pin_memory=True,
        pin_memory_device=device_str
    )

    def requires_grad_false(param: torch.nn.Parameter) -> torch.nn.Parameter:
        param.requires_grad = False
        return param

    starting_params = {
        name: requires_grad_false(param)
        for name, param in pretrained_model.model.named_parameters()
        if (
                param != pretrained_model.classifier
                and name.split('.')[-1] != 'bias'  # It should not be a bias
                and param.requires_grad  # It should be updatable
        )
    }

    sp_regularizer = LSquareStartingPointRegularization(
        starting_parameters=starting_params,
        coefficient=hyperparams.weight_decay_pretrain,
        device=device
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [
            {
                'params': [
                    param for name, param in model_wrapper.model.named_parameters()
                    if param != model_wrapper.classifier
                ],
            },
            {
                'params': [
                    param for name, param in model_wrapper.model.named_parameters()
                    if param == model_wrapper.classifier
                ],
                'weight_decay': hyperparams.weight_decay,
            },
        ],
        lr=hyperparams.learning_rate,
        momentum=hyperparams.momentum
    )

    train(
        model=model_wrapper.model,
        criterion=criterion,
        sp_regularizer=sp_regularizer,
        optimizer=optimizer,
        hyperparams=hyperparams,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
        checkpoint_path=model_path
    )
