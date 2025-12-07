"""
Maze-Generierungs- und Lösungsalgorithmen.
Implementiert DFS, Randomized Prim, BFS und A* mit Visualisierungsunterstützung.
"""
from typing import Generator, Dict, List, Tuple, Optional, Callable
from maze_core import (
    neighbor_dirs, open_wall_index, random_direction, bfs_expand, 
    initialize_maze, DIR_MAP, in_bounds, reset_visited, h_euk, h_man,
    open_random_walls
)
from maze_core import VIS, N, W, E, S
import numpy as np
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Non-interactive Backend, interferiert nicht mit Pygame
import matplotlib.pyplot as plt
import heapq
import os


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


import random
from typing import Dict, Generator, Tuple

import numpy as np

def prim(start_x: int, start_y: int, maze: np.ndarray, measure_mode: bool = False) -> Generator[Dict, None, None]:
    """Randomized Prim's Maze-Generator mit effizienter Frontier-Verwaltung.
    
    Generiert perfekte Mazes durch Frontier-Expansion.
    
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

    # Frontier als Liste initialisieren
    _, initial_frontier = neighbor_dirs(start_x, start_y, maze, visited=False)
    frontier_list: list[Tuple[int, int]] = []
    in_frontier = np.zeros((h, w), dtype=bool)

    for fx, fy in initial_frontier:
        if not maze[fy, fx][VIS] and not in_frontier[fy, fx]:
            frontier_list.append((fx, fy))
            in_frontier[fy, fx] = True
            if not measure_mode:
                yield {"type": "frontier", "pos": (fx, fy)}

    while frontier_list:
        # Wähle zufällige Zelle aus Frontier
        idx = random.randrange(len(frontier_list))
        x, y = frontier_list[idx]

        # Finde Richtungen zu bereits besuchten Nachbarn
        visited_dirs, _ = neighbor_dirs(x, y, maze, visited=True)

        if visited_dirs:
            # Öffne Wand zu einem zufälligen besuchten Nachbarn
            direction = random_direction(visited_dirs)
            dir_idx, opp_idx, dx, dy = open_wall_index(direction)

            nx, ny = x + dx, y + dy

            maze[y, x][dir_idx] = False
            maze[ny, nx][opp_idx] = False
            maze[y, x][VIS] = True

            if not measure_mode:
                yield {"type": "frontierNew", "from": (x, y), "to": (nx, ny)}

            # Füge unbesuchte Nachbarn zur Frontier hinzu
            _, unvisited = neighbor_dirs(x, y, maze, visited=False)
            for ux, uy in unvisited:
                if not maze[uy, ux][VIS] and not in_frontier[uy, ux]:
                    frontier_list.append((ux, uy))
                    in_frontier[uy, ux] = True
                    if not measure_mode:
                        yield {"type": "frontier", "pos": (ux, uy)}

        # Entferne (x, y) aus Frontier
        in_frontier[y, x] = False
        frontier_list[idx] = frontier_list[-1]
        frontier_list.pop()

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
        maze[current[1], current[0], VIS] = 1  # Setze VIS Flag
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
    
    # Generiere n-Werte: 1, step, 2*step, 3*step, ... bis max_n (inklusive)
    n_values_bench = [1] + [step * i for i in range(1, (max_n // step) + 1)]
    
    for alg in algs:
        data: List[List] = []
        
        for n in n_values_bench:
            print(f"\n=== Size {n}×{n} ({alg.__name__}) ===")
            _data = [0.0, 0, 0, 0, 0]  # [avg_len, corridors, nodes, deadends, time]
            
            for _ in range(runs):
                m = initialize_maze(n, n)
                
                # Gemessene Generierung
                start = time.perf_counter()
                deque(alg(0, 0, m, measure_mode=True), maxlen=0)
                elapsed = time.perf_counter() - start
                
                # Analysiere generiertes Maze
                reset_visited(m)
                avg_len, corr_cnt, node_cnt, dead_cnt = analyze_gen(m)
                
                _data[0] += avg_len
                _data[1] += corr_cnt
                _data[2] += node_cnt
                _data[3] += dead_cnt
                _data[4] += elapsed
            
            # Berechne Durchschnittswerte
            avg = [x / runs for x in _data]
            data.append(avg)
            print(f"  {alg.__name__}: {avg[4]:.4f}s, corridors={avg[1]:.0f}, nodes={avg[2]:.0f}, deadends={avg[3]:.0f}, avg_len={avg[0]:.4f}")
        
        plots.append(data)
    
    # Visualisiere Ergebnisse
    n_values = np.array(n_values_bench)
    cells = n_values ** 2
    labels = [alg.__name__ for alg in algs]
    metrics = ["time_s", "corridor_count", "node_count", "deadend_count", "avg_corridor_len"]
    metric_labels = ["Laufzeit (s)", "Anzahl Korridore", "Anzahl Kreuzungen", 
                     "Anzahl Sackgassen", "Ø Korridorlänge"]
    
    # Stil-Variationen für bessere Unterscheidbarkeit
    markers = ['o', 's', '^', 'D', 'v', '<']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # Erstelle einen gemeinsamen Plot mit allen Metriken
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle("Maze-Generierungs-Algorithmen: Vergleich", fontsize=18, fontweight='bold', y=0.995)
    
    # Mapping: Anzeigereihenfolge → Datenindex im _data [avg_len, corridors, nodes, deadends, time]
    data_indices = [4, 1, 2, 3, 0]  # [time, corridors, nodes, deadends, avg_len]
    
    for display_idx in range(5):
        ax = axes.flatten()[display_idx]
        data_idx = data_indices[display_idx]
        for i, (series, label) in enumerate(zip(plots, labels)):
            y = [row[data_idx] for row in series]
            ax.plot(cells, y, 
                   marker=markers[i % len(markers)], 
                   linestyle=linestyles[i % len(linestyles)],
                   label=label, 
                   linewidth=2.5, 
                   markersize=7,
                   alpha=0.85)
        
        ax.set(title=metric_labels[display_idx], xlabel="Zellen (n²)", ylabel=metric_labels[display_idx])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Letztes Subplot ausblenden (nur 5 Metriken)
    axes.flatten()[5].set_visible(False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.99])  # Platz für Suptitle
    fig.subplots_adjust(hspace=0.20, wspace=0.25)  # Moderater Abstand zwischen Subplots
    fig.canvas.draw()  # Force rendering before show
    fig.canvas.flush_events()  # Process GUI events
    
    # Speichere Gesamtplot
    output_dir = "output_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    combined_filename = os.path.join(output_dir, "benchmark_gen_complete.png")
    fig.savefig(combined_filename, dpi=150, bbox_inches='tight')
    print(f"Gesamtplot gespeichert: {combined_filename}")
    
    # Speichere einzelne Subplots
    for display_idx in range(5):
        ax = axes.flatten()[display_idx]
        # Erstelle Figure für einzelne Achse
        fig_single = plt.figure(figsize=(10, 6))
        ax_single = fig_single.add_subplot(111)
        
        data_idx = data_indices[display_idx]
        for i, (series, label) in enumerate(zip(plots, labels)):
            y = [row[data_idx] for row in series]
            ax_single.plot(cells, y, 
                   marker=markers[i % len(markers)], 
                   linestyle=linestyles[i % len(linestyles)],
                   label=label, 
                   linewidth=2.5, 
                   markersize=7,
                   alpha=0.85)
        
        ax_single.set(title=metric_labels[display_idx], xlabel="Zellen (n²)", ylabel=metric_labels[display_idx])
        ax_single.legend()
        ax_single.grid(True, alpha=0.3)
        
        filename = os.path.join(output_dir, f"benchmark_gen_metric_{display_idx+1}_{metric_labels[display_idx].replace(' ', '_').replace('(', '').replace(')', '')}.png")
        fig_single.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Subplot gespeichert: {filename}")
        plt.close(fig_single)
    
    print("\n=== Alle Plots gespeichert in output_plots/ ===")


def analyze_solve(maze: np.ndarray) -> None:
    """Analysiert Lösungs-Eigenschaften eines Maze (Placeholder).
    
    Zukünftige Implementation: Pfadlänge, Effizienz und Explorationsmetriken.
    
    Args:
        maze: Zu analysierendes Maze-Array.
    """
    print("Analyze solve: Implementation pending")




def benchmark_sol(max_n: int, step: int, runs: int, num_walls_list: List[int] = None) -> None:
    """Benchmarkt Maze-Lösungsalgorithmen (BFS, A*) über verschiedene Größen.
    
    Misst Performance-Metriken für alle Solver-Algorithmen auf DFS-generierten Labyrinthen:
    - Algorithmen: BFS, A*_Manhattan, A*_Euclidean, A*_Manhattan_w1.5
    - Metriken: Laufzeit, erkundete Zellen, Pfadlänge, exploration_ratio, optimal_path_ratio, cells_explored_ratio
    - Szenarien: Original-Maze + Maze mit konfigurierbaren geöffneten Wänden
    - Durchschnittswerte über mehrere Läufe pro Konfiguration
    - Verbesserte Seed-Strategie für größenunabhängige Vergleichbarkeit
    """
    def solve_algorithm(alg_name: str, start: Tuple, end: Tuple, maze: np.ndarray) -> Tuple[List, int]:
        """Führt einen Solver-Algorithmus aus und gibt Pfad + erkundete Zellen zurück."""
        path_list: List = []
        
        if alg_name == "BFS":
            deque(bfs(start[0], start[1], maze, end[0], end[1], 
                     measure_mode=True, path_list=path_list), maxlen=0)
        elif alg_name == "A*_Manhattan":
            deque(A_star(start[0], start[1], maze, end[0], end[1], 
                       h_man, 1.0, measure_mode=True, path_list=path_list), maxlen=0)
        elif alg_name == "A*_Euclidean":
            deque(A_star(start[0], start[1], maze, end[0], end[1], 
                       h_euk, 1.0, measure_mode=True, path_list=path_list), maxlen=0)
        else:  # A*_Manhattan_w1.5
            deque(A_star(start[0], start[1], maze, end[0], end[1], 
                       h_man, 1.5, measure_mode=True, path_list=path_list), maxlen=0)
        
        explored = np.count_nonzero(maze[:,:,VIS])
        return path_list, explored
    
    # Algorithmen
    alg_names = ["BFS", "A*_Manhattan", "A*_Euclidean", "A*_Manhattan_w1.5"]
    results: Dict[int, Dict[int, Dict[str, List]]] = {}
    wall_scenarios = [0, 2, 5, 10, 50, 100]  # HIER: Wall-Szenarien definieren - Plots passen sich an
    
    # Generiere n-Werte: step, 2*step, 3*step, ... bis max_n (inklusive)
    n_values_sol = [step * i for i in range(1, (max_n // step) + 1)]
    
    # Benchmarking Loop
    for n in n_values_sol:
        results[n] = {}
        print(f"\n=== Size {n}×{n} ===")
        
        for run_idx in range(runs):
            np.random.seed(42 + run_idx)  # Seed unabhängig von n für Vergleichbarkeit
            
            # Generiere Maze EINMAL pro Größe und Run
            m = initialize_maze(n, n)
            deque(dfs(0, 0, m, measure_mode=True), maxlen=0)
            reset_visited(m)
            
            start, end = (0, 0), (n - 1, n - 1)
            
            # Teste alle Wall-Szenarien auf DIESEM Maze
            for num_walls in wall_scenarios:
                if num_walls == 100:
                    m = initialize_maze(n, n, arr=[False,False,False,False,False])

                if num_walls not in results[n]:
                    results[n][num_walls] = {name: [] for name in alg_names}
                
                # Kopie des Maze für dieses Wall-Szenario
                m_test = m.copy()
                if num_walls > 0:
                    deque(open_random_walls(m_test, int(n*n*(num_walls/100)*4), measure_mode=True), maxlen=0)
                
                # Berechne optimalen Pfad mit BFS als Referenz
                m_ref = m_test.copy()
                reset_visited(m_ref)
                optimal_path_list: List = []
                deque(bfs(start[0], start[1], m_ref, end[0], end[1], 
                         measure_mode=True, path_list=optimal_path_list), maxlen=0)
                optimal_path_len = len(optimal_path_list) if optimal_path_list else 1
                total_cells = n * n
                
                for alg_name in alg_names:
                    reset_visited(m_test)
                    t_start = time.perf_counter()
                    path, explored = solve_algorithm(alg_name, start, end, m_test)
                    elapsed = time.perf_counter() - t_start
                    path_len = len(path) if path else 0
                    
                    # Neue Metriken
                    exploration_ratio = explored / max(path_len, 1)  # Overhead der Exploration
                    optimal_path_ratio = path_len / optimal_path_len if optimal_path_len > 0 else 1.0
                    cells_explored_ratio = explored / total_cells  # Anteil erkundet
                    
                    results[n][num_walls][alg_name].append([
                        elapsed, explored, path_len, exploration_ratio, 
                        optimal_path_ratio, cells_explored_ratio
                    ])
        
        # Durchschnittsbildung nach allen Runs
        for num_walls in wall_scenarios:
            print(f"  walls={num_walls}%")
            for alg_name in alg_names:
                data = results[n][num_walls][alg_name]
                avg = [np.mean([row[i] for row in data]) for i in range(6)]
                results[n][num_walls][alg_name] = avg
                print(f"    {alg_name}: {avg[0]:.4f}s, {avg[1]:.0f} cells, {avg[2]:.1f} path, "
                      f"ratio={avg[4]:.3f}, explored={avg[5]:.2%}")
    
    # Plot-Daten sammeln: Ein Plot pro Wall-Szenario (beliebig erweiterbar)
    plot_scenarios = []
    for num_walls in wall_scenarios:
        plot_data = {name: [] for name in alg_names}
        n_values = []
        for n in sorted(results.keys()):
            if num_walls in results[n]:
                n_values.append(n)
                for alg_name in alg_names:
                    plot_data[alg_name].append(results[n][num_walls][alg_name])
        if n_values:
            title = f"≈ {num_walls} % der Wände geöffnet" if num_walls > 0 else "Original Maze (0 Wände)"
            plot_scenarios.append((plot_data, n_values, title, num_walls))
    
    # Plotting
    metrics = ["time_s", "explored_cells", "path_length", "exploration_ratio", 
               "optimal_path_ratio", "cells_explored_ratio"]
    labels = ["Laufzeit (s)", "Erkundete Zellen", "Pfadlänge", "Exploration Ratio",
              "Pfad/Optimal Ratio", "Erkundete Zellen (%)"]
    
    # Stil-Variationen für bessere Unterscheidbarkeit
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    for plot_data, n_values, title, num_walls in plot_scenarios:
        cells = np.array(n_values) ** 2
        print(f"\n[PLOT] {title}")
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        fig.suptitle(f"Solver-Vergleich: {title}", fontsize=18, fontweight='bold', y=0.995)
        
        for m_idx, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes.flatten()[m_idx]
            for i, alg_name in enumerate(alg_names):
                y = [row[m_idx] for row in plot_data[alg_name]]
                ax.plot(cells, y, 
                       marker=markers[i % 4], 
                       linestyle=linestyles[i % 4],
                       label=alg_name, 
                       linewidth=2.5, 
                       markersize=7,
                       alpha=0.85)
            
            # Spezielle Formatierung für Prozent-Metriken
            if metric == "cells_explored_ratio":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            ax.set(title=label, xlabel="Zellen (n²)", ylabel=label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout(rect=[0, 0, 1, 0.99])  # Platz für Suptitle
        fig.subplots_adjust(hspace=0.20, wspace=0.25)  # Moderater Abstand zwischen Subplots
        fig.canvas.draw()  # Force rendering before show
        fig.canvas.flush_events()  # Process GUI events
        
        # Speichere Gesamtplot
        output_dir = "output_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        wall_str = f"{num_walls}percent" if num_walls > 0 else "0_original"
        combined_filename = os.path.join(output_dir, f"benchmark_sol_walls_{wall_str}_complete.png")
        fig.savefig(combined_filename, dpi=150, bbox_inches='tight')
        print(f"Gesamtplot gespeichert: {combined_filename}")
        plt.close(fig)
        
        # Speichere einzelne Subplots
        for m_idx, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes.flatten()[m_idx]
            # Erstelle Figure für einzelne Achse
            fig_single = plt.figure(figsize=(10, 6))
            ax_single = fig_single.add_subplot(111)
            
            for i, alg_name in enumerate(alg_names):
                y = [row[m_idx] for row in plot_data[alg_name]]
                ax_single.plot(cells, y, 
                       marker=markers[i % 4], 
                       linestyle=linestyles[i % 4],
                       label=alg_name, 
                       linewidth=2.5, 
                       markersize=7,
                       alpha=0.85)
            
            # Spezielle Formatierung für Prozent-Metriken
            if metric == "cells_explored_ratio":
                ax_single.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            ax_single.set(title=label, xlabel="Zellen (n²)", ylabel=label)
            ax_single.legend()
            ax_single.grid(True, alpha=0.3)
            
            wall_str = f"{num_walls}percent" if num_walls > 0 else "0_original"
            filename = os.path.join(output_dir, f"benchmark_sol_walls_{wall_str}_metric_{m_idx+1}_{label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.png")
            fig_single.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Subplot gespeichert: {filename}")
            plt.close(fig_single)
    
    print("\n=== Alle Plots gespeichert in output_plots/ ===")
