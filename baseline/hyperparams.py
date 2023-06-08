from dataclasses import dataclass


@dataclass
class BaseHyperparams:
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
