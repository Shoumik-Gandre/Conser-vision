from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Iterable
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mixins.baseline import BasicEvalStepMixin


@dataclass
class TrainerArgs:
    epochs: int
    batch_size: int
    model_dir: str


def l2_norm_with_starting_point(w: Mapping[str, torch.Tensor], w_sp: Mapping[str, torch.Tensor],
                                device: torch.device) -> torch.Tensor:
    """This function computes the l2 norm of w by considering w_sp as an origin.
    ||w - w_sp||^2_2"""
    param_names = set(w.keys()).intersection(w_sp.keys())
    result = torch.tensor(0.0, dtype=torch.float, device=device)
    for param in param_names:
        result += ((w[param] - w_sp[param]) ** 2).sum()

    return result


class LSquareStartingPointRegularization(torch.nn.Module):
    """This is the L^2-SP regularization from the paper
    Explicit Inductive Bias for Transfer Learning with Convolutional Networks"""

    def __init__(self, pretrained_model: torch.nn.Module, param_names: Iterable[str], device: torch.device):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.param_names = param_names
        self.device = device

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        result = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for param in self.param_names:
            result += ((model.state_dict()[param] - self.pretrained_model.state_dict()[param]) ** 2).sum()

        return result


def l2sp_train_step(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader[Mapping[str, torch.Tensor]],
        pretrained_model: torch.nn.Module,
        l2sp_lambda: float,
        device: torch.device = torch.device('cpu'),
) -> float:
    model.train(True)
    running_loss = 0.0
    pretrained_params = {
        name: param
        for name, param in pretrained_model.named_parameters()
        if name.split('.')[0] != 'fc' and name.split('.')[-1] != 'bias' and param.requires_grad
    }
    for batch in tqdm(dataloader, desc="training"):
        x = batch['image'].to(device)
        y = batch['label'].to(device)

        a = model(x)
        transfer_params = {
            name: param
            for name, param in model.named_parameters()
            if name.split('.')[0] != 'fc' and name.split('.')[-1] != 'bias' and param.requires_grad
        }
        l2_sp = l2_norm_with_starting_point(transfer_params, pretrained_params, device)
        loss = criterion(a, y) + l2sp_lambda * l2_sp

        with torch.no_grad():
            running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(dataloader)


@dataclass
class L2SPTrainer(BasicEvalStepMixin):
    model: torch.nn.Module
    pretrained_model: torch.nn.Module
    l2sp_lambda: float
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.CrossEntropyLoss
    device: torch.device

    def train(self, train_args: TrainerArgs, train_dataset: Dataset, eval_dataset: Dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=train_args.batch_size)
        self.model.to(device=self.device)
        self.pretrained_model.to(device=self.device)

        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = False

        for epoch in range(train_args.epochs):
            print(f"Epoch [{epoch + 1}/{train_args.epochs}]")
            train_loss = l2sp_train_step(
                model=self.model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                dataloader=train_dataloader,
                pretrained_model=self.pretrained_model,
                device=self.device,
                l2sp_lambda=self.l2sp_lambda
            )
            print(train_loss)
            eval_loss = self.eval_step(eval_dataloader)
            print(eval_loss)

        torch.save(self.model, train_args.model_dir)
