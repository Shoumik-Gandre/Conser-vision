"""The train function for l2sp transfer is defined here.
The save checkpoint function and load checkpoint function for l2sp is defined here
"""
import pathlib
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.l2sp.hyperparams import L2SPHyperparams
from src.l2sp.l2sp_regularizer import LSquareStartingPointRegularization


def save_checkpoint(
        checkpoint_path: pathlib.Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float
):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


def load_checkpoint(
        checkpoint_path: pathlib.Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def train(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        sp_regularizer: LSquareStartingPointRegularization,
        optimizer: torch.optim.Optimizer,
        hyperparams: L2SPHyperparams,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        device: torch.device = torch.device('cpu'),
        save_checkpoint_per_epoch: int = 1,
        checkpoint_path: str = ''
):
    for epoch in range(1, hyperparams.num_epochs + 1):
        print(f'epoch [{epoch}/{hyperparams.num_epochs}]')

        # Train step
        model.train(True)
        running_loss = 0.0
        for batch in tqdm(train_dataloader, desc="training"):
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
        print(f"train loss = {running_loss / len(train_dataloader)}")

        # Eval Step
        model.eval()
        running_loss = 0.0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            x = batch['image'].to(device)
            y = batch['label'].to(device)

            with torch.no_grad():
                a = model(x)
                loss = criterion(a, y)
                running_loss += loss.item()

        print(f"eval loss = {running_loss / len(eval_dataloader)}")

        if (epoch - 1) % save_checkpoint_per_epoch == 0:
            save_checkpoint(pathlib.Path(checkpoint_path), model, optimizer, epoch, running_loss)
