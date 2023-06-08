import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

from baseline.hyperparams import BaseHyperparams
from dataset import ImagesDataset
from baseline.trainer import BaseTrainer
import torch

from enumerations import Architectures
from models import get_model


def train_baseline(
        features_csv: str,
        labels_csv: str,
        images_dir: str,
        model_path: str,
        model_arch: Architectures,
        hyperparams: BaseHyperparams
) -> None:
    train_features = pd.read_csv(features_csv, index_col="id")
    train_features['filepath'] = train_features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))
    train_labels = pd.read_csv(labels_csv, index_col="id")

    y = train_labels
    x = train_features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_wrapper = get_model(model_arch)
    train_dataset = ImagesDataset(x_train, y_train, model_wrapper.transforms)
    eval_dataset = ImagesDataset(x_eval, y_eval, model_wrapper.transforms)

    trainer = BaseTrainer(
        model=model_wrapper.model,
        optimizer=torch.optim.AdamW(
            model_wrapper.model.parameters(),
            lr=hyperparams.learning_rate,
            weight_decay=hyperparams.weight_decay,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device
    )
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=hyperparams.num_epochs,
        batch_size=hyperparams.batch_size,
        checkpoint_dir=model_path
    )
