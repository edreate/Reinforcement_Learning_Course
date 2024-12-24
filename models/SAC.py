import torch
from numpy.typing import NDArray
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn.functional as F


class BaseActorNetwork(nn.Module, ABC):
    def __init__(self):
        """
        Base class for all actor networks. Enforces the implementation of `get_action`.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of units in hidden layers.
        """
        super(BaseActorNetwork, self).__init__()

    @abstractmethod
    def get_action(self, state: torch.Tensor) -> NDArray[np.float32]:
        """
        Abstract method to get an action for inference.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            NDArray[np.float32]: Action to take, formatted as a numpy array.
        """
        pass


class ActorNetwork(BaseActorNetwork):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Actor Network for policy prediction.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of units in hidden layers.
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log standard deviation of the actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)  # Limit log_std for numerical stability
        return mean, log_std

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using the reparameterization trick.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled action and log probability of the action.
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Enforce action bounds
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)  # Stability adjustment
        return action, log_prob

    def get_action(self, state: torch.Tensor) -> NDArray[np.float32]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Enforce action bounds
        return action.squeeze(0).numpy()
