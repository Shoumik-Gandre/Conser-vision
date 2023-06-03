from dataclasses import dataclass
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mixins.baseline import BasicTrainStepMixin, BasicEvalStepMixin


@dataclass
class TrainerArgs:
    epochs: int
    batch_size: int
    output_dir: str


@dataclass
class BaselineTrainer(BasicTrainStepMixin, BasicEvalStepMixin):
    model: torch.nn.Module
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
