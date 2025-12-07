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

def export_img(cached_surface, filename: str = None) -> bool:
    """Exportiert das zwischengespeicherte Maze-Rendering als Bilddatei.
    
    Konvertiert die pygame Surface aus der Game-Klasse in ein PIL-Image
    und speichert es als PNG-Datei. Der Dateiname enthält automatisch
    die Größe des Mazes (z.B. maze_50x50.png).
    
    Args:
        cached_surface: pygame.Surface Objekt aus Game.cached_surface.
        filename: Optionaler Ausgabedateiname. Wenn None, wird automatisch
                  generiert als "maze_WIDTHxHEIGHT.png".
    
    Returns:
        True wenn erfolgreich, False bei Fehler.
    
    Raises:
        TypeError: Wenn cached_surface nicht vom Typ pygame.Surface ist.
        IOError: Wenn Datei nicht geschrieben werden kann.
    """
    if cached_surface is None:
        print("Fehler: cached_surface ist None. Bitte erst ein Maze rendern.")
        return False
    
    try:
        import pygame
        
        # Konvertiere pygame Surface zu PIL Image
        raw_str = pygame.image.tobytes(cached_surface, "RGB")
        width, height = cached_surface.get_size()
        pil_image = Image.frombytes("RGB", (width, height), raw_str)
        
        # Generiere Dateiname mit Maze-Größe wenn nicht angegeben
        if filename is None:
            # Berechne Maze-Größe aus Surface-Größe (pixel_size ≈ cell_size)
            maze_size = int(width)  # Approximation der Mazegröße
            filename = f"maze_{maze_size}x{maze_size}.png"
        
        # Speichere als PNG
        pil_image.save(filename)
        print(f"Maze erfolgreich exportiert: {filename}")
        return True
        
    except ImportError:
        print("Fehler: pygame ist nicht installiert.")
        return False
    except Exception as e:
        print(f"Fehler beim Export: {str(e)}")
        return False


def maze_to_img(maze: np.ndarray, cell_size: int = 10, wall_color: Tuple[int, int, int] = (0, 0, 0), 
                path_color: Tuple[int, int, int] = (255, 255, 255), filename: str = None, 
                solve_path: List[Tuple[int, int]] = None, solve_color: Tuple[int, int, int] = (255, 0, 0)) -> bool:
    """Konvertiert ein Maze-Array direkt in eine PNG-Bilddatei.
    
    Malt das Maze als Bild, wobei Wände als schwarze Linien und Pfade als 
    weiße Flächen dargestellt werden. Nutzt DIR_MAP für konsistente Richtungsverarbeitung.
    
    Args:
        maze: Maze-Array der Form (height, width, 5).
        cell_size: Pixelgröße pro Zelle (Standard: 10).
        wall_color: RGB-Farbe für Wände (Standard: schwarz = (0, 0, 0)).
        path_color: RGB-Farbe für Pfade (Standard: weiß = (255, 255, 255)).
        filename: Optionaler Ausgabedateiname. Wenn None, wird automatisch
                  generiert als "maze_HEIGHTxWIDTH.png".
        solve_path: Optionale Liste von (x, y) Koordinaten des Lösungspfads.
        solve_color: RGB-Farbe für den Lösungspfad (Standard: rot = (255, 0, 0)).
    
    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    if maze is None or maze.size == 0:
        print("Fehler: Maze ist leer oder None.")
        return False
    
    try:
        h, w = maze.shape[:2]
        
        # Berechne Bildgröße
        img_width = w * cell_size + 1
        img_height = h * cell_size + 1
        
        # Erstelle Bild (Pfade-Farbe als Hintergrund)
        pil_image = Image.new("RGB", (img_width, img_height), path_color)
        pixels = pil_image.load()
        
        # Male Zellen und Wände mittels DIR_MAP
        for y in range(h):
            for x in range(w):
                px = x * cell_size
                py = y * cell_size
                
                # Nutze DIR_MAP für konsistente Verarbeitung aller Richtungen
                for direction, (dx, dy, _, wall_idx) in DIR_MAP.items():
                    if maze[y, x][wall_idx]:  # Wand existiert
                        if direction == "N":  # Nord-Wand
                            for i in range(cell_size + 1):
                                pixels[px + i, py] = wall_color
                        elif direction == "E":  # Ost-Wand
                            for i in range(cell_size + 1):
                                pixels[px + cell_size, py + i] = wall_color
                        elif direction == "S":  # Süd-Wand
                            for i in range(cell_size + 1):
                                pixels[px + i, py + cell_size] = wall_color
                        elif direction == "W":  # West-Wand
                            for i in range(cell_size + 1):
                                pixels[px, py + i] = wall_color
        
        # Male Lösungspfad falls vorhanden
        if solve_path:
            colored_cells = set()
            colored_connections = set()
            
            for i, (x, y) in enumerate(solve_path):
                px = x * cell_size
                py = y * cell_size
                
                # Fülle Zelle nur wenn noch nicht gefärbt
                if (x, y) not in colored_cells:
                    for dx in range(1, cell_size):
                        for dy in range(1, cell_size):
                            pixels[px + dx, py + dy] = solve_color
                    colored_cells.add((x, y))
                
                # Male Verbindung zur nächsten Zelle
                if i < len(solve_path) - 1:
                    next_x, next_y = solve_path[i + 1]
                    connection = ((x, y), (next_x, next_y))
                    
                    # Färbe Verbindung nur wenn noch nicht gefärbt
                    if connection not in colored_connections:
                        diff_x, diff_y = next_x - x, next_y - y
                        
                        if diff_x == 1:  # Osten
                            for dy in range(1, cell_size):
                                pixels[px + cell_size, py + dy] = solve_color
                        elif diff_x == -1:  # Westen
                            for dy in range(1, cell_size):
                                pixels[px, py + dy] = solve_color
                        elif diff_y == 1:  # Süden
                            for dx in range(1, cell_size):
                                pixels[px + dx, py + cell_size] = solve_color
                        elif diff_y == -1:  # Norden
                            for dx in range(1, cell_size):
                                pixels[px + dx, py] = solve_color
                        
                        colored_connections.add(connection)
        
        # Generiere Dateiname wenn nicht angegeben
        if filename is None:
            filename = f"maze_{h}x{w}.png"
        
        # Speichere als PNG
        pil_image.save(filename)
        print(f"Maze erfolgreich als Bild exportiert: {filename}")
        return True
        
    except Exception as e:
        print(f"Fehler beim Maze-zu-Bild-Export: {str(e)}")
        return False