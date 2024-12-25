from typing import Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from models.base import BaseActorNetwork


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
        Performs a forward pass through the actor network.

        Args:
            state (torch.Tensor): Input state tensor with shape (batch_size, state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Mean of actions (batch_size, action_dim).
                - Log standard deviation of actions (batch_size, action_dim).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)  # Limit log_std for numerical stability
        return mean, log_std

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using the reparameterization trick for exploration.

        Args:
            state (torch.Tensor): Input state tensor with shape (batch_size, state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Sampled action tensor (batch_size, action_dim).
                - Log probability tensor of the action (batch_size, 1).
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
        """
        Generates an action in a deterministic manner for inference.

        Args:
            state (torch.Tensor): Input state tensor with shape (state_dim,).

        Returns:
            NDArray[np.float32]: Action as a numpy array of shape (action_dim,).
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Enforce action bounds
        return action.squeeze(0).numpy()


class SACLosses:
    def __init__(
        self,
        alpha: float = 0.2,
        automatic_entropy: bool = True,
        target_entropy: float = -1.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes the loss computation and entropy coefficient handling.

        Args:
            alpha (float): Initial entropy coefficient.
            automatic_entropy (bool): Whether to use automatic entropy adjustment.
            target_entropy (float): Target entropy value.
        """
        self.automatic_entropy = automatic_entropy
        if automatic_entropy:
            self.log_alpha = torch.tensor(
                [torch.log(torch.tensor(alpha))],
                requires_grad=True,
                device=device,
            )
            self.target_entropy = target_entropy
        else:
            self.alpha = alpha

    def get_alpha(self) -> torch.Tensor:
        """
        Returns the current entropy coefficient used for regularization.

        Returns:
            torch.Tensor: Scalar tensor representing the entropy coefficient.
        """
        return self.log_alpha.exp() if self.automatic_entropy else torch.tensor(self.alpha)

    def alpha_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for updating the entropy coefficient (alpha).

        Args:
            log_prob (torch.Tensor): Log probabilities of actions with shape (batch_size, 1).

        Returns:
            torch.Tensor: Scalar tensor representing the entropy loss.
        """
        if self.automatic_entropy:
            return -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return torch.tensor(0.0, device=log_prob.device)


def compute_actor_loss(
    actor: Type[nn.Module],
    critic_1: Type[nn.Module],
    critic_2: Type[nn.Module],
    states: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the actor loss for training the policy.

    Args:
        actor (ActorNetwork): Actor network for policy prediction.
        critic_1 (CriticNetwork): First critic network.
        critic_2 (CriticNetwork): Second critic network.
        states (torch.Tensor): Batch of input states with shape (batch_size, state_dim).
        alpha (torch.Tensor): Entropy coefficient.

    Returns:
        torch.Tensor: Scalar tensor representing the actor loss.
    """
    actions, log_probs = actor.sample_action(states)
    q1 = critic_1(states, actions)
    q2 = critic_2(states, actions)
    q_min = torch.min(q1, q2)
    return (alpha * log_probs - q_min).mean()


def compute_critic_loss(
    critic: Type[nn.Module],
    target_value_net: Type[nn.Module],
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Computes the critic loss using the Bellman error.

    Args:
        critic (CriticNetwork): Critic network.
        target_value_net (ValueNetwork): Target value network.
        states (torch.Tensor): Batch of states with shape (batch_size, state_dim).
        actions (torch.Tensor): Batch of actions with shape (batch_size, action_dim).
        rewards (torch.Tensor): Batch of rewards with shape (batch_size, 1).
        next_states (torch.Tensor): Batch of next states with shape (batch_size, state_dim).
        dones (torch.Tensor): Batch of done flags with shape (batch_size, 1).
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Scalar tensor representing the critic loss.
    """
    with torch.no_grad():
        target_value = target_value_net(next_states)
        target_q = rewards + gamma * (1 - dones) * target_value
    predicted_q = critic(states, actions)
    return nn.MSELoss()(predicted_q, target_q)


def compute_value_loss(
    actor: Type[nn.Module],
    value_net: Type[nn.Module],
    critic_1: Type[nn.Module],
    critic_2: Type[nn.Module],
    states: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the value loss using the Bellman equation.

    Args:
        actor (ActorNetwork): Actor network for sampling actions.
        value_net (ValueNetwork): Value network.
        critic_1 (CriticNetwork): First critic network.
        critic_2 (CriticNetwork): Second critic network.
        states (torch.Tensor): Batch of states with shape (batch_size, state_dim).
        alpha (torch.Tensor): Entropy coefficient.

    Returns:
        torch.Tensor: Scalar tensor representing the value loss.
    """
    with torch.no_grad():
        actions, log_probs = actor.sample_action(states)
        q1 = critic_1(states, actions)
        q2 = critic_2(states, actions)
        q_min = torch.min(q1, q2)
        target_value = q_min - alpha * log_probs
    predicted_value = value_net(states)
    return nn.MSELoss()(predicted_value, target_value)
