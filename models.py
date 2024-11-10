import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1: nn.Linear = nn.Linear(n_observations, 128)
        self.layer2: nn.Linear = nn.Linear(128, 128)
        self.layer3: nn.Linear = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DQN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_observations)

        Returns:
            Tensor: Output tensor of shape (batch_size, n_actions)
        """

        # Input x shape: (batch_size, n_observations)
        x = F.relu(self.layer1(x))
        # After layer1: (batch_size, 128)
        x = F.relu(self.layer2(x))
        # After layer2: (batch_size, 128)
        x = self.layer3(x)
        # After layer3 (output layer): (batch_size, n_actions)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()

        self.l1 = nn.Linear(n_observations, 128, bias=False)
        self.l2 = nn.Linear(128, n_actions, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = nn.Dropout(p=0.6)(x)
        x = nn.ReLU()(x)
        x = self.l2(x)
        action_probs = nn.Softmax(dim=-1)(x)
        return action_probs
