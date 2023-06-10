from typing import Mapping

import torch


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
            result += torch.norm(model.state_dict()[param_name].to(self.device) - param.to(self.device))

        return self.coefficient * result
