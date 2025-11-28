
# Refactored graphics: thin wrapper with same public API
from maze_ui.colors import Color
from maze_ui.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from maze_ui.runtime import init_pygame
from maze_ui.render import draw_cell, draw_maze_pygame, init_empty
from maze_ui.app.game import Game
