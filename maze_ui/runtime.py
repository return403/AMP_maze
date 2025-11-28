
"""Pygame und pygame_gui Initialisierung."""
from typing import Tuple
import pygame
import pygame_gui
import numpy as np
from .constants import SCREEN_WIDTH, SCREEN_HEIGHT


def init_pygame(maze: np.ndarray) -> Tuple[pygame.time.Clock, pygame.Surface, pygame_gui.UIManager]:
    """Initialisiere Pygame und pygame_gui für die Anwendung.
    
    Setzt Fenster, Display-Manager und Event-Clock auf.
    
    Args:
        maze: Referenz-Maze (aktuell ungenutzt, für zukünftige Erweiterung).
    
    Returns:
        Tuple mit drei Komponenten:
        - clock: Pygame Clock für FPS-Control.
        - screen: Pygame Surface für Rendering.
        - manager: pygame_gui UIManager für UI-Elemente.
    """
    pygame.init()
    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Generator/Solver")
    clock = pygame.time.Clock()
    
    return clock, screen, manager
