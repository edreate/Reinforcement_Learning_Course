from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


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
            state (torch.Tensor): Input state tensor with shape (state_dim,).

        Returns:
            NDArray[np.float32]: Action to take as a numpy array of shape (action_dim,).
        """
        pass
