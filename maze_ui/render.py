
"""Rendering-Funktionen für Maze-Visualisierung auf Pygame-Surfaces."""
from typing import Optional
import pygame
import numpy as np
from maze_core import N, E, S, W, initialize_maze
from .colors import Color


def draw_cell(
    screen: pygame.Surface,
    maze: np.ndarray,
    x: int,
    y: int,
    cell_size: int,
    color: tuple,
) -> None:
    """Zeichne eine einzelne Maze-Zelle mit ihren Wänden.
    
    Renderiert Zellenfüllung und schwarze Linien für existierende Wände
    (Nord, Ost, Süd, West).
    
    Args:
        screen: Pygame-Surface zum Zeichnen.
        maze: Maze-Array mit Wandinformationen.
        x: X-Koordinate der Zelle.
        y: Y-Koordinate der Zelle.
        cell_size: Größe einer Zelle in Pixeln.
        color: RGB-Farbtuple für Zellfüllung.
    """
    cx, cy = x * cell_size, y * cell_size
    cell = maze[y, x]
    
    # Zelle füllen
    pygame.draw.rect(screen, color, (cx, cy, cell_size, cell_size))
    
    # Wände zeichnen (schwarz wenn vorhanden)
    if cell[N]:
        pygame.draw.line(
            screen, Color.BLACK.value,
            (cx, cy), (cx + cell_size, cy)
        )
    if cell[E]:
        pygame.draw.line(
            screen, Color.BLACK.value,
            (cx + cell_size, cy), (cx + cell_size, cy + cell_size)
        )
    if cell[S]:
        pygame.draw.line(
            screen, Color.BLACK.value,
            (cx, cy + cell_size), (cx + cell_size, cy + cell_size)
        )
    if cell[W]:
        pygame.draw.line(
            screen, Color.BLACK.value,
            (cx, cy), (cx, cy + cell_size)
        )

def draw_maze(
    screen: pygame.Surface,
    maze: np.ndarray,
    cell_size: int,
    color: tuple = None,
) -> None:
    """Zeichne das gesamte Maze.
    
    Iteriert über alle Zellen und zeichnet jede mit draw_cell.
    
    Args:
        screen: Pygame-Surface zum Zeichnen.
        maze: Maze-Array zur Visualisierung.
        cell_size: Größe jeder Zelle in Pixeln.
        color: Zellfüllfarbe (Standard: Weiß).
    """
    if color is None:
        color = Color.WHITE.value
    
    h, w = maze.shape[:2]
    for y in range(h):
        for x in range(w):
            draw_cell(screen, maze, x, y, cell_size, color)

# Alias für Kompatibilität
draw_maze_pygame = draw_maze


def init_empty(
    width: int,
    height: int,
    maze: np.ndarray,
    cell_size: int,
    screen: pygame.Surface,
    heat_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Erstellt ein leeres Maze (alle Wände offen) und visualisiert es.
    Markiert Heatmap-Bereiche farblich.
    
    Args:
        width, height: Maze-Dimensionen
        maze: Referenz-Maze (nicht verwendet, aber erwartet)
        cell_size: Zellgröße in Pixeln
        screen: Pygame-Surface zum Zeichnen
        heat_map: Optional Heatmap zur Farbmarkierung
    
    Returns:
        Neues leeres Maze-Array
    """
    new_maze = initialize_maze(width, height, [False, False, False, False, False])
    
    for y in range(height):
        for x in range(width):
            # Zellen mit Heatmap-Wert 0 weiß, sonst Teal
            if heat_map is not None and heat_map[y, x] == 0:
                draw_cell(screen, new_maze, x, y, cell_size, Color.WHITE.value)
            else:
                draw_cell(screen, new_maze, x, y, cell_size, Color.TEAL.value)
    
    return new_maze
