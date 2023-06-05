import pathlib

import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split
from dataset import ImagesDataset
from l2sp.trainer import TrainerArgs, L2SPTrainer
from l2sp.model import get_l2sp_model
import torch


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
    trainer = L2SPTrainer(
        model=model,
        optimizer=torch.optim.Adam(
            [
                {
                    'params': [param for name, param in model.named_parameters() if name.split('.')[0] != 'fc'],
                    'lr': 1e-5,
                },
                {
                    'params': [param for name, param in model.named_parameters() if name.split('.')[0] == 'fc'],
                    'lr': 1e-3,
                    'weight_decay': 1e-3,
                },
            ],
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        pretrained_model=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT),
    )
    trainer.train(train_args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
