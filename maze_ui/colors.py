
"""Farbpalette für Maze-Visualisierung und Algorithmus-Schritt-Färbung."""
from enum import Enum


class Color(Enum):
    """Vordefinierte Farben für UI und Algorithmus-Visualisierung als RGB-Tuples."""
    
    # Primärfarben
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    
    # Sekundärfarben
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    
    # Spezialfarben
    ORANGE = (255, 165, 0)
    TEAL = (0, 206, 206)
    
    # Basis
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
