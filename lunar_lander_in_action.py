
import torch
import gymnasium as gym
import pygame
import numpy as np
from pathlib import Path
from typing import Tuple, Type


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

    def overlay_info(self, text: str, position: Tuple[int, int] = (10, 10), color: Tuple[int, int, int] = (255, 255, 255)):
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


def load_model(filename: Path, model_class: Type[torch.nn.Module], n_observations: int, n_actions: int) -> torch.nn.Module:
    """
    Loads a model of the specified class with given parameters.
    
    Args:
        filename (Path): Path to the saved model file.
        model_class (Type[torch.nn.Module]): The class of the model to load.
        n_observations (int): Number of observations (input features).
        n_actions (int): Number of actions (output features).

    Returns:
        torch.nn.Module: Loaded model instance.
    """
    try:
        model = model_class(n_observations, n_actions)
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


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

import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    MODEL_FILE_PATH = Path("output/policy_network_lunar_lander_v3_bs_16_2024-12-01_17-03-50_very_good.pth")
    MODEL_CLASS = MLP  # Replace with DQN or other model class as needed

    # Initialize the environment to get observation and action space sizes
    env = gym.make("LunarLander-v3")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    policy_net = load_model(MODEL_FILE_PATH, MODEL_CLASS, n_observations, n_actions)
    run_lunar_lander(policy_net)


    env.close()

