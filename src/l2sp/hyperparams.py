from dataclasses import dataclass


@dataclass
class L2SPHyperparams:
    learning_rate: float
    weight_decay_pretrain: float
    weight_decay: float
    momentum: float
    batch_size: int
    num_epochs: int
