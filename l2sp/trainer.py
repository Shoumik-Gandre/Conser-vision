from dataclasses import dataclass
from typing import Mapping, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from l2sp.l2sp_regularizer import LSquareStartingPointRegularization
from mixins.baseline import BasicEvalStepMixin


@dataclass
class TrainerArgs:
    epochs: int
    batch_size: int
    model_dir: str


@dataclass
class LSquareStartingPointHyperparameters:
    pretrain_coefficient: float
    param_names: Iterable[str]


def l2sp_train_step(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader[Mapping[str, torch.Tensor]],
        sp_regularizer: LSquareStartingPointRegularization,
        device: torch.device = torch.device('cpu'),
) -> float:
    model.train(True)
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="training"):
        x = batch['image'].to(device)
        y = batch['label'].to(device)

        a = model(x)
        loss = criterion(a, y)

        with torch.no_grad():
            running_loss += loss.item()

        loss += sp_regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(dataloader)


@dataclass
class L2SPTrainer(BasicEvalStepMixin):
    model: torch.nn.Module
    sp_regularizer: LSquareStartingPointRegularization
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.CrossEntropyLoss
    device: torch.device

    def train(self, train_args: TrainerArgs, train_dataset: Dataset, eval_dataset: Dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=train_args.batch_size)
        self.model.to(device=self.device)

        for epoch in range(train_args.epochs):
            print(f"Epoch [{epoch + 1}/{train_args.epochs}]")
            train_loss = l2sp_train_step(
                model=self.model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                dataloader=train_dataloader,
                sp_regularizer=self.sp_regularizer,
                device=self.device,
            )
            print(train_loss)
            eval_loss = self.eval_step(eval_dataloader)
            print(eval_loss)

        torch.save(self.model, train_args.model_dir)
