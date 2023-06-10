import pathlib
import typing
from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mixins.baseline import BasicPredictStepMixin
from src.dataset import ImagesDataset
from src.enumerations import Architectures
from src.models import load_model


def predict_animal(
        model_path: str,
        features_csv: str,
        images_dir: str,
        prediction_path: str,
        architecture: Architectures,
        batch_size: int,
        device: torch.device
) -> None:
    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))

    x = features.filepath.to_frame()

    model, transforms = load_model(model_path, architecture, device)
    dataset = ImagesDataset(x, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds_collector = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # run the forward step
            logits = model.forward(batch["image"].to(device))
            # apply softmax so that model outputs are in range [0,1]
            preds = torch.nn.functional.softmax(logits, dim=1)
            # store this batch's predictions in df
            preds_df = pd.DataFrame(
                preds.detach().cpu().numpy(),
                index=batch["image_id"],
                columns=['antelope_duiker',
                         'bird',
                         'blank',
                         'civet_genet',
                         'hog',
                         'leopard',
                         'monkey_prosimian',
                         'rodent'],
            )
            preds_collector.append(preds_df)

    submission_df = pd.concat(preds_collector)
    submission_df.to_csv(prediction_path)


@dataclass
class BaselinePredictor(BasicPredictStepMixin):
    model: torch.nn.Module
    device: torch.device


def predict(
        model_path: str,
        features_csv: str,
        images_dir: str,
        prediction_path: str,
        architecture: Architectures,
        batch_size: int,
        device: torch.device
) -> None:
    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))
    x = features['filepath'].to_frame()

    model, transforms = load_model(model_path, architecture, device)
    dataset = ImagesDataset(x, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    predictions = get_predictions(dataloader, model, device)
    store_predictions(path=pathlib.Path(prediction_path), predictions=predictions)


def get_predictions(dataloader: DataLoader[Mapping[str, torch.Tensor]],
                    model: torch.nn.Module, device: torch.device) -> typing.Mapping[str, list]:
    """Uses the model to predict on the dataloader and accumulates a mapping of predictions"""

    predictions: typing.MutableMapping[str, typing.List[typing.Any]] = {
        'image_id': [],
        'probabilities': []
    }
    model.eval()

    for batch in tqdm(dataloader, desc="prediction"):
        x = batch['image'].to(device)
        batch_ids = batch['image_id']

        with torch.no_grad():
            a = model(x)
            a = torch.nn.functional.softmax(a, dim=-1)
            a = a.detach().cpu().numpy()

        predictions['image_id'].append(batch_ids)
        predictions['probabilities'].append(a)

    return predictions


def store_predictions(path: pathlib.Path, predictions: Mapping[str, typing.List]) -> None:
    dataframe = pd.DataFrame(
        predictions['probabilities'],
        index=predictions['image_id'],
        columns=['antelope_duiker',
                 'bird',
                 'blank',
                 'civet_genet',
                 'hog',
                 'leopard',
                 'monkey_prosimian',
                 'rodent'
                 ],
    )
    dataframe.to_csv(path)
