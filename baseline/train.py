import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import ImagesDataset
from baseline.trainer import TrainerArgs, BaselineTrainer
from baseline.model import get_baseline_model
import torch


def train_baseline(train_features_csv: str, train_labels_csv: str, train_images_dir: str, output_dir: str) -> None:
    train_features = pd.read_csv(train_features_csv, index_col="id")
    train_features['filepath'] = train_features['filepath'].apply(
        lambda path: pathlib.Path(train_images_dir) / str(path))
    train_labels = pd.read_csv(train_labels_csv, index_col="id")

    y = train_labels
    x = train_features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImagesDataset(x_train, y_train)
    eval_dataset = ImagesDataset(x_eval, y_eval)
    training_args = TrainerArgs(epochs=2, batch_size=32, output_dir=output_dir)
    model = get_baseline_model()
    trainer = BaselineTrainer(
        model=model,
        optimizer=torch.optim.Adam(
            [
                {
                    'params': [param for name, param in model.named_parameters() if name.split('.')[0] != 'fc'],
                    'lr': 1e-4
                },
                {
                    'params': [param for name, param in model.named_parameters() if name.split('.')[0] == 'fc'],
                    'lr': 1e-3,
                },
            ],
            lr=1e-3,
            weight_decay=1e-2
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device
    )
    trainer.train(train_args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
