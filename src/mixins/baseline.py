from typing import Protocol, Mapping, List, Any, MutableMapping

import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class HasBasicTrainAttributes(Protocol):

    @property
    def model(self) -> torch.nn.Module: return ...
    @property
    def criterion(self) -> torch.nn.Module: return ...
    @property
    def optimizer(self) -> torch.optim.Optimizer: return ...
    @property
    def device(self) -> torch.device: return ...


class HasBasicEvalAttributes(Protocol):

    @property
    def model(self) -> torch.nn.Module: return ...
    @property
    def criterion(self) -> torch.nn.Module: return ...
    @property
    def device(self) -> torch.device: return ...


class HasBasicPredictAttributes(Protocol):

    @property
    def model(self) -> torch.nn.Module: return ...
    @property
    def device(self) -> torch.device: return ...


class BasicTrainStepMixin:

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


class BasicEvalStepMixin:

    def eval_step(self: HasBasicEvalAttributes, dataloader: DataLoader[Mapping[str, torch.Tensor]]) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="evaluation"):
            x = batch['image'].to(self.device)
            y = batch['label'].to(self.device)

            with torch.no_grad():
                a = self.model(x)
                loss = self.criterion(a, y)
                total_loss += loss.item()

        return total_loss / len(dataloader)


class BasicPredictStepMixin:
    """A mixin to define a simple prediction step for conservision competition.
    Make sure that the child class has the following attributes:
    1. model of type torch.nn.Module
    2. device of type torch.device"""

    def prediction_step(self: HasBasicPredictAttributes, dataloader: DataLoader[Mapping[str, torch.Tensor]]
                        ) -> Mapping[str, List[Any]]:
        predictions: MutableMapping[str, List[Any]] = {
            'image_id': [],
            'probabilities': []
        }
        self.model.eval()

        for batch in tqdm(dataloader, desc="prediction"):
            x = batch['image'].to(self.device)
            batch_ids = batch['image_id']

            with torch.no_grad():
                a = self.model(x)
                a = torch.nn.functional.softmax(a, dim=1)
                a = a.detach().cpu().numpy()

            predictions['image_id'].append(batch_ids)
            predictions['probabilities'].append(a)

        return predictions


