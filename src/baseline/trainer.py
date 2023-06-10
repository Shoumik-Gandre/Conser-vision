import pathlib
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset

from mixins.baseline import BasicTrainStepMixin, BasicEvalStepMixin


@dataclass
class BaseTrainer(BasicTrainStepMixin, BasicEvalStepMixin):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.CrossEntropyLoss
    device: torch.device

    def train(self, train_dataset: Dataset, eval_dataset: Dataset, epochs: int, batch_size: int, checkpoint_dir: str):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.model.to(device=self.device)

        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            train_loss = self.train_step(train_dataloader)
            print(f"training loss   = {train_loss:.4f}")
            eval_loss = self.eval_step(eval_dataloader)
            print(f"validation loss = {eval_loss:.4f}")

            torch.save(self.model, pathlib.Path(checkpoint_dir) / f'epoch-{epoch}.pt')
