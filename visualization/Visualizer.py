from typing import Tuple

import numpy as np
import pygame
from numpy.typing import NDArray


class PygameVisualizer:
    """
    A simple Pygame-based visualizer.
    This class handles initialization, rendering of frames, displaying text overlays,
    and updating the Pygame window for smooth visualization.
    """

    def __init__(
        self,
        caption: str = "Visualization",
        screen_width: int = 1200,
        screen_height: int = 800,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        text_position: Tuple[int, int] = (10, 10),
    ):
        """
        Initializes the PygameVisualizer with basic display settings.

        :param caption: Window title.
        :param screen_width: Width of the display window in pixels.
        :param screen_height: Height of the display window in pixels.
        :param text_color: RGB color tuple for text rendering.
        :param text_position: (x, y) position for text overlay.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = None
        self.clock = None
        self.font = None
        self.caption = caption
        self.text_colour = text_color
        self.text_position = text_position

    def initialize(self) -> None:
        """
        Initializes the Pygame environment and sets up the main screen
        and font for rendering text.
        """
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def overlay_info(self, text: str) -> None:
        """
        Renders a text overlay on the screen.

        :param text: The text string to display.
        """
        text_surface = self.font.render(text, True, self.text_colour)
        self.screen.blit(text_surface, self.text_position)

    def render_frame(self, frame: NDArray[np.int8]) -> None:
        """
        Renders a single frame (in the form of a NumPy array) onto the
        Pygame window.

        :param frame: A 3D NumPy array representing an RGB image.
        """
        # Transpose from (height, width, channels) to (width, height, channels)
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled_surface = pygame.transform.scale(frame_surface, (self.screen_width, self.screen_height))
        self.screen.blit(scaled_surface, (0, 0))

    def update_display(self) -> None:
        """
        Updates the Pygame display to show the latest rendered frame
        and any text overlays.
        """
        pygame.display.flip()

    def tick(self, fps: int = 30) -> None:
        """
        Regulates the frame rate of the visualization.

        :param fps: Target frames per second.
        """
        self.clock.tick(fps)

    @staticmethod
    def quit() -> None:
        """
        Quits and uninitializes the Pygame environment, closing any
        open windows.
        """
        pygame.quit()
