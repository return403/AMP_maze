"""
Maze-Generierungs- und Lösungsalgorithmen.
Implementiert DFS, Randomized Prim, BFS und A* mit Visualisierungsunterstützung.
"""
from typing import Generator, Dict, List, Tuple, Optional, Callable
from maze_core import (
    neighbor_dirs, open_wall_index, random_direction, bfs_expand, 
    initialize_maze, DIR_MAP, in_bounds, reset_visited, h_euk, h_man
)
from maze_core import VIS, N, W, E, S
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import heapq


# ===== Maze-Generierungsalgorithmen =====

def dfs(start_x: int, start_y: int, maze: np.ndarray, measure_mode: bool = False) -> Generator[Dict, None, None]:
    """Depth-First-Search Maze-Generator mit Stack-basiertem Backtracking.
    
    Generiert ein perfektes Maze durch Tiefensuche mit Wandöffnung und Rückverfolgung.
    Erzeugt Events für Start, Bewegung, Rückverfolgung und Fertigstellung.
    
    Args:
        start_x: X-Koordinate des Startpunkts.
        start_y: Y-Koordinate des Startpunkts.
        maze: Maze-Array der Form (height, width, 5) mit Wandinformationen.
        measure_mode: True für reine Zeitmessungen ohne Visualisierungsereignisse.
    
    Yields:
        Dict mit Visualisierungsereignissen:
        - {"type": "start", "pos": (x, y)}
        - {"type": "forward", "from": (x, y), "to": (nx, ny)}
        - {"type": "backtrack", "from": (x, y), "to": (x, y)}
        - {"type": "done", "pos": (x, y)}
    """
    h, w = maze.shape[:2]
    stack: List[Tuple[int, int]] = [(start_x, start_y)]
    maze[start_y, start_x][VIS] = True
    
    if not measure_mode:
        yield {"type": "start", "pos": (start_x, start_y)}
    
    while stack:
        x, y = stack[-1]
        dirs, _ = neighbor_dirs(x, y, maze, visited=False)
        
        if not dirs:
            # Keine unbesuchten Nachbarn -> Backtrack
            old_pos = (x, y)
            stack.pop()
            if stack:
                new_pos = stack[-1]
                if not measure_mode:
                    yield {"type": "backtrack", "from": old_pos, "to": new_pos}
            else:
                if not measure_mode:
                    yield {"type": "done", "pos": (start_x, start_y)}
                break
        else:
            # Wähle zufällige Richtung und öffne Wand
            direction = random_direction(dirs)
            dir_idx, opp_idx, dx, dy = open_wall_index(direction)
            maze[y, x][dir_idx] = False
            nx, ny = x + dx, y + dy
            maze[ny, nx][opp_idx] = False
            maze[ny, nx][VIS] = True
            stack.append((nx, ny))
            if not measure_mode:
                yield {"type": "forward", "from": (x, y), "to": (nx, ny)}


def prim(start_x: int, start_y: int, maze: np.ndarray, measure_mode: bool = False) -> Generator[Dict, None, None]:
    """Randomized Prim's Maze-Generator mit Frontier-Set-Verwaltung.
    
    Generiert perfekte Mazes durch Frontier-Expansion, ideal für große Mazes.
    Verwaltet aktive Grenzregion mit zufälliger Wandöffnung.
    
    Args:
        start_x: X-Koordinate des Startpunkts.
        start_y: Y-Koordinate des Startpunkts.
        maze: Maze-Array der Form (height, width, 5) mit Wandinformationen.
        measure_mode: True für reine Zeitmessungen ohne Visualisierungsereignisse.
    
    Yields:
        Dict mit Visualisierungsereignissen:
        - {"type": "start", "pos": (x, y)}
        - {"type": "frontier", "pos": (x, y)}
        - {"type": "frontierNew", "from": (x, y), "to": (nx, ny)}
        - {"type": "done", "pos": (x, y)}
    """
    h, w = maze.shape[:2]
    maze[start_y, start_x][VIS] = True
    
    if not measure_mode:
        yield {"type": "start", "pos": (start_x, start_y)}
    
    # Initialisiere Frontier mit Nachbarn des Startpunkts
    _, frontier_list = neighbor_dirs(start_x, start_y, maze, visited=False)
    frontier: set = set(frontier_list)
    
    while frontier:
        # Wähle zufällige Zelle aus Frontier
        x, y = random_direction(list(frontier))
        visited_dirs, _ = neighbor_dirs(x, y, maze, visited=True)
        
        if not visited_dirs:
            # Keine besuchten Nachbarn -> entferne aus Frontier
            frontier.remove((x, y))
            continue
        
        # Öffne Wand zu einem zufälligen besuchten Nachbarn
        direction = random_direction(visited_dirs)
        dir_idx, opp_idx, dx, dy = open_wall_index(direction)
        maze[y, x][dir_idx] = False
        nx, ny = x + dx, y + dy
        maze[ny, nx][opp_idx] = False
        maze[y, x][VIS] = True
        
        if not measure_mode:
            yield {"type": "frontierNew", "from": (x, y), "to": (nx, ny)}
        
        # Füge unbesuchte Nachbarn zur Frontier hinzu
        _, unvisited = neighbor_dirs(x, y, maze, visited=False)
        for ux, uy in unvisited:
            if not maze[uy, ux][VIS] and (ux, uy) not in frontier:
                frontier.add((ux, uy))
                if not measure_mode:
                    yield {"type": "frontier", "pos": (ux, uy)}
        
        frontier.remove((x, y))
    
    if not measure_mode:
        yield {"type": "done", "pos": (start_x, start_y)}



# ===== Maze-Lösungsalgorithmen =====

def bfs(start_x: int, start_y: int, maze: np.ndarray, end_x: int, end_y: int, 
        measure_mode: bool = False, path_list: Optional[List] = None) -> Generator[Dict, None, None]:
    """Breadth-First-Search Maze-Solver mit Distanz-basierten Visualisierung.
    
    Findet kürzeste Pfade in ungewichteten Mazes. Nutzt Distanz-Tracking für
    Gradient-Visualisierung während der Exploration.
    
    Args:
        start_x: X-Koordinate des Startpunkts.
        start_y: Y-Koordinate des Startpunkts.
        maze: Maze-Array zur Lösung.
        end_x: X-Koordinate des Zielpunkts.
        end_y: Y-Koordinate des Zielpunkts.
        measure_mode: True für reine Zeitmessungen ohne Visualisierungsereignisse.
        path_list: Liste zum Speichern des gefundenen Pfades (wird erweitert).
    
    Yields:
        Dict mit Visualisierungsereignissen:
        - {"type": "start", "pos": (x, y)}
        - {"type": "BFS", "pos": (x, y), "value": distance}
        - {"type": "Path", "pos": (x, y)}
        - {"type": "done_bfs", "pos": (x, y)}
    """
    if path_list is None:
        path_list = []
    
    h, w = maze.shape[:2]
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    distances = np.full((h, w), -1, dtype=int)
    distances[start_y, start_x] = 0
    maze[start_y, start_x][VIS] = True
    
    if not measure_mode:
        yield {"type": "start", "pos": (start_x, start_y)}
    
    queue: deque = deque([(start_x, start_y)])
    
    while queue:
        x, y = queue.popleft()
        if x == end_x and y == end_y:
            break
        
        if not measure_mode:
            yield {"type": "BFS", "pos": (x, y), "value": distances[y, x]}
        
        # Erweitere Queue mit Nachbarn
        neighbors, new_parents = bfs_expand(x, y, maze, distances)
        if neighbors:
            queue.extend(neighbors)
        parents.update(new_parents)
    
    # Rekonstruiere Pfad
    path: List[Tuple[int, int]] = [(end_x, end_y)]
    while path[-1] != (start_x, start_y):
        path.append(parents[path[-1]])
    path.reverse()
    path_list.extend(path)
    
    if not measure_mode:
        for p in path:
            yield {"type": "Path", "pos": p}
        yield {"type": "done_bfs", "pos": (end_x, end_y)}



def A_star(start_x: int, start_y: int, maze: np.ndarray, end_x: int, end_y: int, 
           h_mode: Callable = h_man, weight: float = 1.0, heat_maps: Optional[np.ndarray] = None, 
           measure_mode: bool = False, path_list: Optional[List] = None) -> Generator[Dict, None, None]:
    """A* Pathfinding mit Heuristic-Support und Heatmap-Integration.
    
    Effiziente Lösungssuche mit konfigurierbaren Heuristics (Manhattan, Euklidisch).
    Unterstützt gewichtete Pfade durch Heatmaps.
    
    Args:
        start_x: X-Koordinate des Startpunkts.
        start_y: Y-Koordinate des Startpunkts.
        maze: Zu lösendes Maze-Array.
        end_x: X-Koordinate des Zielpunkts.
        end_y: Y-Koordinate des Zielpunkts.
        h_mode: Heuristic-Funktion (h_man für Manhattan, h_euk für Euklidisch).
        weight: Gewichtungsfaktor der Heuristic (1.0=optimal, >1=schneller,suboptimal).
        heat_maps: Optionale Heatmap für gewichtete Kosten.
        measure_mode: True für reine Zeitmessungen ohne Visualisierungsereignisse.
        path_list: Liste zum Speichern des gefundenen Pfades (wird erweitert).
    
    Yields:
        Dict mit Visualisierungsereignissen:
        - {"type": "start", "pos": (x, y)}
        - {"type": "open", "pos": (x, y)}
        - {"type": "close", "pos": (x, y)}
        - {"type": "done_bfs", "pos": (x, y)}
    """
    if path_list is None:
        path_list = []
    
    start: Tuple[int, int] = (start_x, start_y)
    end: Tuple[int, int] = (end_x, end_y)
    h, w = maze.shape[:2]
    
    # Heatmap-Verarbeitung
    if heat_maps is None:
        heat_maps = np.zeros((h, w), dtype=np.int64)
    elif isinstance(heat_maps, tuple):
        # Mehrere Heatmaps summieren
        heat_maps = np.sum(np.stack(heat_maps), axis=0)
    
    # A* State-Variablen
    g_scores: Dict[Tuple[int, int], float] = {start: 0}
    open_list = []
    closed_list: set = set()
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    # Initialisiere Open-List mit Start
    h0 = h_mode(start_x, start_y, end_x, end_y) * weight
    heapq.heappush(open_list, (h0, h0, start))
    
    if not measure_mode:
        yield {"type": "start", "pos": start}
    
    while open_list:
        _, _, current = heapq.heappop(open_list)
        
        # Ziel erreicht
        if current == end:
            path: List[Tuple[int, int]] = [end]
            while current != start:
                current = parents[current]
                path.append(current)
            path.reverse()
            path_list.extend(path)
            return path
        
        # Skip wenn bereits besucht
        if current in closed_list:
            continue
        
        closed_list.add(current)
        if not measure_mode:
            yield {"type": "close", "pos": current}
        
        x, y = current
        
        # Exploriere alle Nachbarn
        for direction in DIR_MAP:
            dx, dy, _, wall_idx = DIR_MAP[direction]
            nx, ny = x + dx, y + dy
            neighbor: Tuple[int, int] = (nx, ny)
            
            # Überspringe ungültige oder blockierte Nachbarn
            if not in_bounds(nx, ny, w, h):
                continue
            if maze[y, x][wall_idx]:  # Wand blockiert
                continue
            if neighbor in closed_list:
                continue
            
            # Berechne neue Kosten
            new_g = g_scores[current] + 1 + heat_maps[ny, nx]
            
            # Überspringe wenn schlechterer Pfad
            if neighbor in g_scores and new_g >= g_scores[neighbor]:
                continue
            
            # Update beste Route
            parents[neighbor] = current
            g_scores[neighbor] = new_g
            h_value = h_mode(nx, ny, end_x, end_y) * weight
            f_value = new_g + h_value
            
            heapq.heappush(open_list, (f_value, h_value, neighbor))
            if not measure_mode:
                yield {"type": "open", "pos": neighbor}
    
    # Kein Pfad gefunden
    if not measure_mode:
        yield {"type": "done_bfs", "pos": end, "status": "no_path"}
    return None



# ===== Maze-Analyse =====

def analyze_gen(maze: np.ndarray) -> Tuple[float, int, int, int]:
    """Analysiert strukturelle Eigenschaften eines generierten Maze.
    
    Berechnet durchschnittliche Korridorlänge, Korridoranzahl, Kreuzungen und
    Sackgassen durch DFS-Traversal und Topologie-Klassifizierung.
    
    Args:
        maze: Zu analysierendes Maze-Array.
    
    Returns:
        Tuple mit vier Metriken:
        - avg_corridor_length: Durchschnittliche Länge der Korridore (float).
        - corridor_count: Anzahl der Korridorsegmente (int).
        - node_count: Anzahl der Kreuzungen, 3+ Verbindungen (int).
        - deadend_count: Anzahl der Sackgassen (int).
    """
    h, w = maze.shape[:2]
    stack: List[Tuple[int, int]] = [(0, 0)]
    corridor_lengths: List[int] = []
    current_corridor = 0
    corridor_count = 0
    node_count = 0
    deadend_count = 0
    
    while stack:
        x, y = stack.pop()
        if maze[y, x][VIS]:
            continue
        
        maze[y, x][VIS] = True
        
        # Zähle offene Nachbarn
        open_neighbors: List[Tuple[int, int]] = []
        for direction in DIR_MAP:
            nx, ny = x + DIR_MAP[direction][0], y + DIR_MAP[direction][1]
            if in_bounds(nx, ny, w, h) and not maze[y, x][DIR_MAP[direction][3]]:
                open_neighbors.append((nx, ny))
        
        n_open = len(open_neighbors)
        
        # Klassifiziere Zellentyp
        if n_open == 1:
            # Sackgasse
            deadend_count += 1
            if current_corridor > 0:
                corridor_lengths.append(current_corridor)
                corridor_count += 1
                current_corridor = 0
        elif n_open == 2:
            # Korridor
            current_corridor += 1
        elif n_open >= 3:
            # Kreuzung
            node_count += 1
            if current_corridor > 0:
                corridor_lengths.append(current_corridor)
                corridor_count += 1
                current_corridor = 0
        
        # Füge Nachbarn hinzu
        for nx, ny in open_neighbors:
            if not maze[ny, nx][VIS]:
                stack.append((nx, ny))
    
    # Finalisiere letzten Korridor
    if current_corridor > 0:
        corridor_lengths.append(current_corridor)
        corridor_count += 1
    
    avg_len = float(np.mean(corridor_lengths)) if corridor_lengths else 0.0
    return avg_len, corridor_count, node_count, deadend_count


def benchmark(max_n: int, step: int, runs: int, maze: np.ndarray, algs: List[Callable]) -> None:
    """Benchmarkt Maze-Generierungsalgorithmen und visualisiert Ergebnisse.
    
    Testet Algorithmen über verschiedene Größen, misst Strukturmetriken und
    Laufzeiten, visualisiert Ergebnisse mit matplotlib-Plots.
    
    Args:
        max_n: Maximale Labyrinth-Größe n für n×n Mazes.
        step: Schrittweite zwischen aufeinanderfolgenden Größenmessungen.
        runs: Anzahl der Wiederholungen pro Größe für Durchschnittsbildung.
        maze: Template Maze (aktuell ungenutzt, für zukünftige Erweiterung).
        algs: Liste der Generierungsalgorithmen zum Benchmarken.
    """
    plots: List[List] = []
    
    for alg in algs:
        data: List[List] = []
        
        for n in range(1, max_n, step):
            print(f"Testing {alg.__name__} at size {n}×{n}...")
            accumulator = [0.0, 0, 0, 0, 0]  # [avg_len, corridors, nodes, deadends, time]
            
            for _ in range(runs):
                m = initialize_maze(n, n)
                
                # Gemessene Generierung
                start = time.perf_counter()
                deque(alg(0, 0, m, measure_mode=True), maxlen=0)
                elapsed = time.perf_counter() - start
                
                # Analysiere generiertes Maze
                reset_visited(m)
                avg_len, corr_cnt, node_cnt, dead_cnt = analyze_gen(m)
                
                accumulator[0] += avg_len
                accumulator[1] += corr_cnt
                accumulator[2] += node_cnt
                accumulator[3] += dead_cnt
                accumulator[4] += elapsed
            
            # Berechne Durchschnittswerte
            data.append([x / runs for x in accumulator])
        
        plots.append(data)
    
    # Visualisiere Ergebnisse
    n_values = np.arange(1, max_n, step)
    cells = n_values ** 2
    labels = [alg.__name__ for alg in algs]
    metrics = ["avg_corridor_len", "corridor_count", "node_count", "deadend_count", "time_s"]
    
    for metric_idx in range(5):
        plt.figure(figsize=(10, 6))
        for series, label in zip(plots, labels):
            y = [row[metric_idx] for row in series]
            plt.plot(cells, y, marker="o", label=label, linewidth=2)
        
        plt.title(f"Metrik: {metrics[metric_idx]}")
        plt.xlabel("Zellen (n²)")
        plt.ylabel(metrics[metric_idx])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.show()


def analyze_solve(maze: np.ndarray) -> None:
    """Analysiert Lösungs-Eigenschaften eines Maze (Placeholder).
    
    Zukünftige Implementation: Pfadlänge, Effizienz und Explorationsmetriken.
    
    Args:
        maze: Zu analysierendes Maze-Array.
    """
    print("Analyze solve: Implementation pending")




