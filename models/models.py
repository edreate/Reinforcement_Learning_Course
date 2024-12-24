import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        Critic Network to predict Q-values.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of units in hidden layers.
        """
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state (torch.Tensor): Input state.
            action (torch.Tensor): Input action.

        Returns:
            torch.Tensor: Predicted Q-value.
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        """
        Value Network to predict the soft value function.

        Args:
            state_dim (int): Dimension of the input state.
            hidden_dim (int): Number of units in hidden layers.
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Predicted value.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)
