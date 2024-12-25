import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DiscretePolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """
        state: torch tensor of shape [batch_size, state_dim]
        returns: a distribution (Categorical) over actions.
        """
        logits = self.net(state)  # shape: [batch_size, action_dim]
        return Categorical(logits=logits)

    def get_action(self, state):
        """
        state: single state, shape [state_dim]
        returns: sampled_action (int), log_prob_of_that_action
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ContinuousPolicyNetwork, self).__init__()

        # Network to output means of each action dimension
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # A separate parameter for log_std (assuming diagonal covariance)
        # Alternatively, you could make this an nn.Linear(...) for each dim
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """
        state: torch tensor of shape [batch_size, state_dim]
        returns: a Normal distribution for the action.
        """
        mean = self.net(state)  # shape: [batch_size, action_dim]
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, state):
        """
        state: single state, shape [state_dim]
        returns: sampled_action (numpy array), log_prob_of_that_action
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum across action dimensions
        return action.squeeze(0).detach().numpy(), log_prob
