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


@dataclass
class LSquareStartingPointHyperparameters:
    pretrain_coefficient: float
    param_names: Iterable[str]


# class LSquareStartingPointRegularization(torch.nn.Module):
#     """This is the L^2-SP regularization from the paper
#     Explicit Inductive Bias for Transfer Learning with Convolutional Networks"""
#
#     def __init__(self,
#                  pretrained_model: torch.nn.Module,
#                  param_names: Iterable[str],
#                  coefficient: float, device: torch.device):
#         super().__init__()
#         self.pretrained_model = pretrained_model
#         self.param_names = param_names
#         self.device = device
#         self.coefficient = coefficient
#
#     def forward(self, model: torch.nn.Module) -> torch.Tensor:
#         result = torch.tensor(0.0, dtype=torch.float, device=self.device)
#         for param in self.param_names:
#             result += ((model.state_dict()[param] - self.pretrained_model.state_dict()[param]) ** 2).sum()
#
#         return self.coefficient * result


class LSquareStartingPointRegularization(torch.nn.Module):
    """This is the L^2-SP regularization from the paper
    Explicit Inductive Bias for Transfer Learning with Convolutional Networks"""

    def __init__(self, starting_parameters: Mapping[str, torch.nn.Module], coefficient: float, device: torch.device):
        super().__init__()
        self.starting_parameters = starting_parameters
        self.device = device
        self.coefficient = coefficient

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        result = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for param_name, param in self.starting_parameters.items():
            result += torch.norm(model.state_dict()[param_name] - param)

        return self.coefficient * result


def l2sp_train_step_(
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
        loss = criterion(a, y)

        with torch.no_grad():
            running_loss += loss.item()

        loss += l2sp_lambda * l2_sp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(dataloader)


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
        # self.pretrained_model.to(device=self.device)

        # def requires_grad_false(param: torch.nn.Parameter) -> torch.nn.Parameter:
        #     param.requires_grad = False
        #     return param
        #
        # starting_params = {
        #     name: requires_grad_false(param)
        #     for name, param in self.pretrained_model.named_parameters()
        #     if (
        #             name.split('.')[0] != 'fc'  # It should not be a classifier
        #             and name.split('.')[-1] != 'bias'  # It should not be a bias
        #             and param.requires_grad  # It should be updatable
        #     )
        # }
        #
        # sp_regularize = LSquareStartingPointRegularization(
        #     starting_parameters=starting_params,
        #     coefficient=1e-2, device=self.device)

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
