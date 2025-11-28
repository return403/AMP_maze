# maze_core.py
"""
Kernfunktionen für Maze-Verwaltung und Algorithmen.
Enthält Hilfsfunktionen für Navigation, Validierung und Bildverarbeitung.
"""
from typing import Tuple, Dict, List, Callable
import numpy as np
from random import choice, randrange
from PIL import Image, ImageOps

# ===== Konstanten für Maze-Struktur =====
# Richtungs-Map: Verknüpft Himmelsrichtung mit (dx, dy, Gegenrichtung, Index)
DIR_MAP = {
    "N": (0, -1, "S", 0),
    "E": (1, 0, "W", 1),
    "S": (0, 1, "N", 2),
    "W": (-1, 0, "E", 3),
}

# Indizes für Zellkomponenten [N, E, S, W, VIS]
N, E, S, W, VIS = 0, 1, 2, 3, 4
CELL_COMPONENTS = {"N": N, "E": E, "S": S, "W": W, "VIS": VIS}


# ===== Maze-Initialisierung =====
def initialize_maze(
    width: int, height: int, arr: List[bool] = None
) -> np.ndarray:
    """Erstellt ein neues Maze-Array mit gegebenen Dimensionen.
    
    Initialisiert ein 3D-Array, wo jede Zelle 5 boolsche Komponenten enthält:
    [N-Wand, E-Wand, S-Wand, W-Wand, Besuchsmarke]
    
    Args:
        width: Breite des Mazes in Zellen.
        height: Höhe des Mazes in Zellen.
        arr: Standard-Zellwerte [5 boolsche Werte] oder None (Standard: alle Wände).
    
    Returns:
        2D NumPy-Array der Form (height, width, 5) mit Datentyp bool.
    """
    if arr is None:
        arr = [True, True, True, True, False]
    return np.array(
        [[arr.copy() for _ in range(width)] for _ in range(height)], 
        dtype=np.bool_
    )


# ===== Koordinaten- und Richtungsvalidierung =====
def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    """Prüft, ob Koordinaten innerhalb der Maze-Grenzen liegen.
    
    Args:
        x: X-Koordinate zu prüfen.
        y: Y-Koordinate zu prüfen.
        width: Breite des Mazes.
        height: Höhe des Mazes.
    
    Returns:
        True wenn Koordinaten gültig, sonst False.
    """
    return 0 <= x < width and 0 <= y < height


def random_direction(options: List[str]) -> str:
    """Wählt zufällig eine Richtung aus einer Liste aus.
    
    Args:
        options: Liste von Richtungsstrings aus ["N", "E", "S", "W"].
    
    Returns:
        Ein zufällig ausgewählter Richtungsstring.
    """
    return choice(options)


def neighbor_dirs(
    x: int, y: int, maze: np.ndarray, visited: bool
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Findet Nachbarzellen mit bestimmtem Besuchsstatus.
    
    Iteriert über alle vier Himmelsrichtungen und gibt Nachbarn zurück,
    die den gewünschten Besuchsstatus haben.
    
    Args:
        x: X-Koordinate der Zelle.
        y: Y-Koordinate der Zelle.
        maze: Maze-Array.
        visited: Besuchsstatus zum Filtern (True=besucht, False=unbesucht).
    
    Returns:
        Tuple von (directions, coordinates):
        - directions: Liste von Richtungsstrings ["N", "E", "S", "W"].
        - coordinates: Liste von (x, y) Tupeln für Nachbarzellen.
    """
    h, w = maze.shape[:2]
    directions, coordinates = [], []
    
    for direction, (dx, dy, _, _) in DIR_MAP.items():
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, w, h) and bool(maze[ny, nx][VIS]) == visited:
            directions.append(direction)
            coordinates.append((nx, ny))
    
    return directions, coordinates


# ===== Zustandsmanagement =====
def reset_visited(maze: np.ndarray) -> None:
    """Setzt alle Besuchsflags des Mazes zurück für Neustarts.
    
    Modifiziert das Maze-Array in-place, setzt Index [4] (VIS) aller Zellen auf False.
    
    Args:
        maze: Maze-Array (wird in-place modifiziert).
    """
    maze[:, :, VIS] = False


# ===== Algorithmus-Hilfsfunktionen =====
def bfs_expand(
    x: int, y: int, maze: np.ndarray, distances: np.ndarray
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]:
    """BFS-Expandierungsschritt für erreichbare Nachbarzellen.
    
    Erweitert die BFS-Frontier von einer Position aus zu allen erreichbaren,
    noch unbesuchten Nachbarn. Verwaltet Parent-Tracking für Rückverfolgung.
    
    Args:
        x: X-Koordinate der Quellzelle.
        y: Y-Koordinate der Quellzelle.
        maze: Maze-Array.
        distances: Distanz-Array (wird aktualisiert für neue Zellen).
    
    Returns:
        Tuple (next_positions, parents):
        - next_positions: Liste neuer erreichbarer Zellkoordinaten.
        - parents: Dict mit Parent-Zuordnungen für Rückverfolgung.
    """
    h, w = maze.shape[:2]
    next_positions = []
    parents = {}
    
    for direction, (dx, dy, opp_dir, _) in DIR_MAP.items():
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, w, h):
            opp_idx = DIR_MAP[opp_dir][3]
            if not maze[ny, nx][VIS] and not maze[ny, nx][opp_idx]:
                maze[ny, nx][VIS] = True
                distances[ny, nx] = distances[y, x] + 1
                parents[(nx, ny)] = (x, y)
                next_positions.append((nx, ny))
    
    return next_positions, parents


def open_wall_index(direction: str) -> Tuple[int, int, int, int]:
    """Gibt Wandindizes für Richtung zurück (zum Öffnen von Wänden).
    
    Berechnet die notwendigen Indizes und Verschiebungsvektoren um eine Wand
    zwischen zwei benachbarten Zellen zu öffnen.
    
    Args:
        direction: Richtungsstring aus ["N", "E", "S", "W"].
    
    Returns:
        Tuple (wall_idx, opp_idx, dx, dy):
        - wall_idx: Wandindex in der aktuellen Zelle.
        - opp_idx: Wandindex in der Nachbarzelle (Gegenseite).
        - dx, dy: Verschiebungsvektoren zur Nachbarzelle.
    """
    wall_idx = DIR_MAP[direction][3]
    opp_direction = DIR_MAP[direction][2]
    opp_idx = DIR_MAP[opp_direction][3]
    dx, dy = DIR_MAP[direction][0], DIR_MAP[direction][1]
    return wall_idx, opp_idx, dx, dy


# ===== Heuristiken für A* =====
def h_euk(x: int, y: int, end_x: int, end_y: int) -> float:
    """Euklidische Heuristic (Luftlinien-Distanz) für A*.
    
    Args:
        x: Aktuelle X-Koordinate.
        y: Aktuelle Y-Koordinate.
        end_x: Ziel X-Koordinate.
        end_y: Ziel Y-Koordinate.
    
    Returns:
        Euklidische Distanz als float.
    """
    return ((end_x - x) ** 2 + (end_y - y) ** 2) ** 0.5


def h_man(x: int, y: int, end_x: int, end_y: int) -> float:
    """Manhattan-Heuristic (Gitterbewegung) für A*.
    
    Args:
        x: Aktuelle X-Koordinate.
        y: Aktuelle Y-Koordinate.
        end_x: Ziel X-Koordinate.
        end_y: Ziel Y-Koordinate.
    
    Returns:
        Manhattan-Distanz als float.
    """
    return abs(end_x - x) + abs(end_y - y)


# Heuristik-Registry
HEURISTICS: Dict[str, Callable[[int, int, int, int], float]] = {
    "h_euk": h_euk,
    "h_man": h_man,
}


def f_n(h_value: float, g_value: float) -> float:
    """A*-Bewertungsfunktion f(n) = h(n) + g(n).
    
    Kombiniert Heuristic-Schätzung mit bisherigen Kosten für optimale Pfadsuche.
    
    Args:
        h_value: Heuristische Schätzung zum Ziel.
        g_value: Bisherige Kosten vom Startpunkt.
    
    Returns:
        Gesamtkostenschätzung für A*-Priorität.
    """
    return h_value + g_value


# ===== Maze-Manipulationen =====
def open_random_walls(
    maze: np.ndarray, count: int, measure_mode: bool = False
):
    """Öffnet zufällig 'count' Wände zwischen benachbarten Zellen.
    
    Erzeugt zusätzliche Pfade in einem generierten Maze zur Erhöhung der Komplexität.
    Kann als Generator für Visualisierung verwendet werden.
    
    Args:
        maze: Maze-Array (wird modifiziert).
        count: Anzahl der zufällig zu öffnenden Wände.
        measure_mode: True für Zeitmessungen ohne Visualisierungsereignisse.
    
    Yields:
        Dict mit Visualisierungsereignissen:
        - {"type": "randomWall", "from": (x, y), "to": (nx, ny)}
        - {"type": "done", "pos": (0, 0)}
    """
    reset_visited(maze)
    h, w = maze.shape[:2]
    
    for _ in range(count):
        x, y = randrange(w), randrange(h)
        directions, _ = neighbor_dirs(x, y, maze, False)
        
        if not directions:
            continue
        
        direction = random_direction(directions)
        wall_idx, opp_idx, dx, dy = open_wall_index(direction)
        nx, ny = x + dx, y + dy
        
        if not in_bounds(nx, ny, w, h):
            continue
        
        # Öffne Wand in beiden Zellen
        maze[y, x][wall_idx] = False
        maze[ny, nx][opp_idx] = False
        
        if not measure_mode:
            yield {"type": "randomWall", "from": (x, y), "to": (nx, ny)}
    
    if not measure_mode:
        yield {"type": "done", "pos": (0, 0)}


# Alias für Kompatibilität
Open_Random_Walls = open_random_walls


# ===== Bildverarbeitung =====
def import_img(
    maze: np.ndarray, weight: float = 1.0, path: str = "bild.jpg", thresh: int = 180
) -> np.ndarray:
    """Konvertiert ein Bild in eine Heatmap mit Maze-Dimensionen.
    
    Lädt ein Bild, konvertiert es zu Graustufen, skaliert auf Maze-Größe,
    invertiert und wendet Schwellwert an um Heatmap zu erzeugen.
    
    Args:
        maze: Referenz-Maze (bestimmt Zielgröße).
        weight: Gewichtungsfaktor für Heatmap-Werte (Hintergrund).
        path: Pfad zur Bilddatei.
        thresh: Schwellwert für Binarisierung (0-255).
    
    Returns:
        2D NumPy-Array (float32) mit Kostengewichtung pro Zelle.
    """
    h, w = maze.shape[:2]
    
    # Bild laden, in Graustufen konvertieren und auf Maze-Größe skalieren
    gray_img = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
    gray_img = ImageOps.invert(gray_img)
    
    # In Array konvertieren und Schwellwert anwenden
    arr = np.array(gray_img, dtype=np.uint8)
    mask = arr >= thresh  # True = helle Linien (niedrige Kosten)
    
    # Linien → 0, Hintergrund → weight
    heatmap = (1 - mask.astype(np.uint8)) * weight
    return heatmap.astype(np.float32)