from dataclasses import dataclass
from typing import Any, Mapping, Protocol
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mixins.baseline import BasicEvalStepMixin, HasBasicTrainAttributes


@dataclass
class TrainerArgs:
    epochs: int
    batch_size: int
    output_dir: str


class HasL2SPTrainAttributes(Protocol):

    @property
    def model(self) -> torch.nn.Module: return ...
    @property
    def criterion(self) -> torch.nn.Module: return ...
    @property
    def optimizer(self) -> torch.optim.Optimizer: return ...
    @property
    def device(self) -> torch.device: return ...
    @property
    def pretrained_model(self) -> torch.nn.Module: return ...


def l2sp_regularize(model: torch.nn.Module, pretrained_model: torch.nn.Module):
    ...


class L2SPTrainStepMixin:

    def train_step(self: HasBasicTrainAttributes, dataloader: DataLoader[Mapping[str, torch.Tensor]]) -> float:
        self.model.train(True)
        running_loss = 0.0
        for batch in tqdm(dataloader, desc="training"):
            x = batch['image'].to(self.device)
            y = batch['label'].to(self.device)

            a = self.model(x)
            loss = self.criterion(a, y)

            with torch.no_grad():
                running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return running_loss / len(dataloader)


@dataclass
class L2SPTrainer(L2SPTrainStepMixin, BasicEvalStepMixin):
    model: torch.nn.Module
    pretrained_weights: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.CrossEntropyLoss
    device: torch.device

    def train(self, train_args: TrainerArgs, train_dataset: Dataset, eval_dataset: Dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=train_args.batch_size)
        self.model.to(device=self.device)

        for epoch in range(train_args.epochs):
            print(f"Epoch [{epoch+1}/{train_args.epochs}]")
            train_loss = self.train_step(train_dataloader)
            print(train_loss)
            eval_loss = self.eval_step(eval_dataloader)
            print(eval_loss)

        torch.save(self.model, train_args.output_dir)
