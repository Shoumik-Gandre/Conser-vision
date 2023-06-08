from dataclasses import dataclass
from yamldataclassconfig.config import YamlDataClassConfig



@dataclass
class BaseHyperparams(YamlDataClassConfig):
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
