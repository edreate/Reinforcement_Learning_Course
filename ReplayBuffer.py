from collections import deque, namedtuple
from typing import Deque, List, Tuple

import numpy as np
from numpy.typing import NDArray

# Define a namedtuple to store experience tuples
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            state_dim (int): Dimensionality of the state.
            action_dim (int): Dimensionality of the action.
        """
        self.capacity: int = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

    def add(self, state: NDArray, action: NDArray, reward: float, next_state: NDArray, done: bool) -> None:
        """
        Adds an experience to the replay buffer.

        Args:
            state (NDArray): The current state.
            action (NDArray): The action taken.
            reward (float): The reward received.
            next_state (NDArray): The next state.
            done (bool): Whether the episode ended after this step.
        """
        assert state.shape == (self.state_dim,), f"State shape must be ({self.state_dim},)"
        assert action.shape == (self.action_dim,), f"Action shape must be ({self.action_dim},)"
        assert next_state.shape == (self.state_dim,), f"Next state shape must be ({self.state_dim},)"

        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]: A batch of experiences.
        """
        assert len(self.buffer) >= batch_size, "Not enough samples in the buffer to sample the batch."

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        experiences: List[Experience] = [self.buffer[idx] for idx in indices]

        # Stack and return the batch
        states = np.array([exp.state for exp in experiences], dtype=np.float32)
        actions = np.array([exp.action for exp in experiences], dtype=np.float32)
        rewards = np.array([exp.reward for exp in experiences], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in experiences], dtype=np.float32)
        dones = np.array([exp.done for exp in experiences], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Returns the current size of the replay buffer.

        Returns:
            int: Number of experiences stored.
        """
        return len(self.buffer)
