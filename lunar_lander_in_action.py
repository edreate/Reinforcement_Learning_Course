"""
Lunar Lander Visualization and Model Execution Script

This script is designed to visualize and run the Lunar Lander v3 environment
using a fixed PolicyNetwork architecture. The PolicyNetwork will be trained
using various reinforcement learning algorithms (off-policy and on-policy),
such as DQN, VPG, PPO, and SAC. The architecture of the PolicyNetwork will
remain consistent across all experiments.

Key Features:
- Pygame-based visualization for rendering the environment.
- Supports loading pre-trained PolicyNetwork models for evaluation.
- Designed for training with multiple RL algorithms while keeping
  the network architecture fixed.
"""

from pathlib import Path
from typing import Tuple, Type

import gymnasium as gym
import numpy as np
import pygame
import torch

from models.models import PolicyNetwork


class LunarLanderVisualizer:
    def __init__(self, screen_width: int = 1200, screen_height: int = 800):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = None
        self.clock = None
        self.font = None

    def initialize(self):
        """Initializes the Pygame environment."""
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Lunar Lander - Model Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def overlay_info(
        self, text: str, position: Tuple[int, int] = (10, 10), color: Tuple[int, int, int] = (255, 255, 255)
    ):
        """Renders text overlay on the screen."""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def render_frame(self, frame: np.ndarray):
        """Renders a single frame onto the Pygame window."""
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.screen.blit(pygame.transform.scale(frame_surface, (self.screen_width, self.screen_height)), (0, 0))

    def update_display(self):
        """Updates the Pygame display."""
        pygame.display.flip()

    def tick(self, fps: int = 30):
        """Maintains the desired frame rate."""
        self.clock.tick(fps)

    @staticmethod
    def quit():
        """Quits the Pygame environment."""
        pygame.quit()


def load_model(filename: Path, n_observations: int, n_actions: int) -> torch.nn.Module:
    """
    Loads a PolicyNetwork with the given parameters.

    Args:
        filename (Path): Path to the saved model file.
        n_observations (int): Number of observations (input features).
        n_actions (int): Number of actions (output features).

    Returns:
        torch.nn.Module: Loaded model instance.
    """
    try:
        model = PolicyNetwork(n_observations, n_actions)
        model.load_state_dict(torch.load(filename, map_location=torch.device("cpu"), weights_only=True))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e


def select_action(state: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Selects an action based on the current state using the trained model."""
    with torch.no_grad():
        return model(state).max(1)[1].view(1, 1)


def run_lunar_lander(model: torch.nn.Module, render_fps: int = 20):
    """
    Runs the Lunar Lander simulation using the provided trained model.

    Args:
        model (torch.nn.Module): Trained model for action selection.
        render_fps (int): Frames per second for rendering.
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    visualizer = LunarLanderVisualizer()
    visualizer.initialize()

    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    timestep = 0
    cumulative_reward = 0

    while not done:
        action = select_action(state_tensor, model)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        cumulative_reward += reward
        timestep += 1

        frame = env.render()
        visualizer.render_frame(frame)

        info_text = (
            f"Timestep: {timestep}      Reward: {cumulative_reward:.2f}     \n"
            f"Lander X: {observation[0]:.2f}       Lander Y: {observation[1]:.2f}     \n"
            f"Velocity X: {observation[2]:.2f}      Velocity Y: {observation[3]:.2f}"
        )
        visualizer.overlay_info(info_text)
        visualizer.update_display()

        done = terminated or truncated
        visualizer.tick(render_fps)

    env.close()
    visualizer.quit()


def benchmark_model(model: torch.nn.Module, n_episodes: int = 100, max_steps: int = 1000):
    """
    Benchmarks the model over multiple episodes and computes average performance metrics.

    Args:
        model (torch.nn.Module): Trained model for action selection.
        n_episodes (int): Number of episodes to run for benchmarking.
        max_steps (int): Maximum steps allowed per episode.

    Returns:
        dict: Average metrics (reward, steps, efficiency).
    """
    env = gym.make("LunarLander-v3")
    total_rewards = []
    total_steps = []
    score_efficiencies = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        cumulative_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = select_action(state_tensor, model)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            cumulative_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(cumulative_reward)
        total_steps.append(steps)
        score_efficiencies.append(cumulative_reward / steps)

        print(f"Episode {episode + 1}/{n_episodes} -> Reward: {cumulative_reward:.2f}, Steps: {steps}")

    env.close()

    # Compute averages
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    avg_efficiency = np.mean(score_efficiencies)

    return {"Average Reward": avg_reward, "Average Steps": avg_steps, "Average Efficiency": avg_efficiency}


if __name__ == "__main__":
    MODEL_FILE_PATH = Path("output/dqn_policy_network_lunar_lander_v3_2024-12-02_19-36-49.pth")

    # Initialize the environment to get observation and action space sizes
    env = gym.make("LunarLander-v3")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    # Load the model
    policy_net = load_model(MODEL_FILE_PATH, n_observations, n_actions)

    # # Run benchmarking
    # metrics = benchmark_model(policy_net, n_episodes=100)
    # print("\nBenchmark Results:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.2f}")

    run_lunar_lander(policy_net)
