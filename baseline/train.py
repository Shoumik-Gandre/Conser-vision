import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import ImagesDataset
from baseline.trainer import TrainerArgs, BaselineTrainer
import torch

from enumerations import Architectures
from models import get_model


def train_baseline(
        features_csv: str,
        labels_csv: str,
        images_dir: str,
        model_path: str,
        model_arch: Architectures,
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

    model, transforms = get_model(model_arch)
    train_dataset = ImagesDataset(x_train, y_train, transforms)
    eval_dataset = ImagesDataset(x_eval, y_eval, transforms)
    training_args = TrainerArgs(epochs=epochs, batch_size=batch_size, output_dir=model_path)

    for name, param in model.named_parameters():
        if name.split('.')[0] != 'fc':
            param.requires_grad = False

    trainer = BaselineTrainer(
        model=model,
        optimizer=torch.optim.AdamW(
            [
                # {
                #     'params': [param for name, param in model.named_parameters() if name.split('.')[0] != 'fc'],
                #     'lr': 1e-4,
                # },
                {
                    'params': [
                        param for name, param in model.named_parameters()
                        if name.split('.')[0] == 'fc' and param.requires_grad
                    ],
                    # 'lr': 1e-3,
                },
            ],
            # lr=1e-3,
            weight_decay=1e-3
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device
    )
    trainer.train(train_args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
