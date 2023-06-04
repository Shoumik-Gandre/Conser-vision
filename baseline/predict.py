import pathlib
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImagesDataset
from baseline.model import load_baseline_model
import torch

from mixins.baseline import BasicPredictStepMixin


def predict_baseline(model_path: str, features_csv: str, images_dir: str, prediction_path: str,
                     batch_size: int, device: torch.device) -> None:
    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))

    x = features.filepath.to_frame()

    dataset = ImagesDataset(x)
    model = load_baseline_model(model_path, device)
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


def predict_baseline2(
        model_path: str,
        features_csv: str,
        images_dir: str,
        prediction_path: str,
        batch_size: int,
        device: torch.device) -> None:
    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_dir) / str(path))

    x = features['filepath'].to_frame()

    dataset = ImagesDataset(x)
    model = load_baseline_model(model_path, device)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    predictions = BaselinePredictor(model, device).prediction_step(dataloader)
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
    dataframe.to_csv(prediction_path)
