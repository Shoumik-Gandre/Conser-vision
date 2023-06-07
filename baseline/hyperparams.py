from dataclasses import dataclass


@dataclass
class BaselineHyperparams:
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
