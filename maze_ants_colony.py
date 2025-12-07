# maze_algorithms.py
from maze_core import neighbor_dirs, open_wall_index, random_direction, bfs_expand, initialize_maze, DIR_MAP, in_bounds, reset_visited, h_euk, f_n, h_man
from maze_core import VIS, N, W, E, S
import numpy as np
import time
from collections import deque
import heapq
from math import inf
from random import random

def ants_colony(start_x, start_y, maze, end_x, end_y,
                maxit=100, ants=50, decay=0.1, heat_maps=None,
                measure_mode=False, path_list=None):
    h, w = maze.shape[:2]
    pheromone_map = np.full((h, w), 0, dtype=float)

    best_path = None  # bisher bester Pfad zur Zielzelle
    decayrate = max(0.001, min(1.0, abs(1 - decay)))

    HEAT_MAP = heat_maps if heat_maps is not None else np.zeros((h, w), dtype=np.float64)

    if not measure_mode:
        yield {"type": "start", "pos": (start_x, start_y)}

    for _ in range(maxit):
        # neue Ameisen für diese Iteration
        ant_list = [Ant((start_x, start_y), (end_x, end_y), maze, decayrate) for _ in range(ants)]
        finished = []
        # alle Ameisen laufen lassen
        while ant_list:
            next_ants = []
            for ant in ant_list:
                old = ant.pos
                cont = ant.move(maze, pheromone_map, HEAT_MAP)
                new = ant.pos

                if not measure_mode:
                    yield {"type": "forward", "from": old, "to": new}

                if cont:
                    # Ameise kann weiterlaufen
                    next_ants.append(ant)
                else:
                    # Ziel erreicht ODER Sackgasse -> als "fertig" zählen
                    finished.append(ant)

            ant_list = next_ants

        # globales Pheromon-Decay
        #pheromone_map *= decayrate

        # Pheromon durch alle "fertigen" Ameisen auftragen
        for ant in finished:
            ant.pathreconstruction()        # Pfad vom Start bis zur aktuellen Position
            ant.place_pheromone(pheromone_map)  # auch Teilpfade verstärken

            # Nur echte Zielpfade als finale Lösung merken
            if ant.goal and ant.path:
                if best_path is None or len(ant.path) < len(best_path):
                    best_path = list(ant.path)

    # Am Ende: falls ein Zielpfad gefunden wurde, ausgeben
    if best_path is not None and not measure_mode:
        for p in best_path:
            yield {"type": "Path", "pos": p}
        yield {"type": "done_bfs", "pos": (end_x, end_y)}

    # Pfad an path_list weitergeben (nur wenn vorhanden und Pfad gefunden)
    if best_path is not None and path_list is not None:
        path_list.extend(best_path)
    return
               
class Ant:
    def __init__(self, start, end, maze, decay):
        self.h, self.w = maze.shape[:2]
        self.parents = {}
        self.alpha = 1.0
        self.beta = 1.0
        self.max_pheromone = 10.0
        self.decayrate = decay
        self.start = start
        self.pos = start
        self.end = end
        self.goal = False
        self.path = []

        self.vis = np.full((self.h, self.w), False, dtype=bool)
        self.vis[self.start[1], self.start[0]] = True

    def rannumber(self):
        return random()

    def move(self, maze, pher, heu):
        # Ziel schon erreicht?
        if self.pos == self.end:
            self.goal = True
            return False  # nicht weiterlaufen

        # mögliche Moves holen
        dirs_, neighbor_cords = self.allowed_moves(maze)
        if not neighbor_cords:
            # Sackgasse
            self.goal = False
            return False

        ex, ey = self.end

        # Attraktivitäten berechnen (Pheromon * Heuristik)
        weights = []
        for (x, y) in neighbor_cords:
            tau = pher[y, x] ** self.alpha

            # Heuristik: nahe am Ziel + niedrige "Heatmap"-Kosten
            dist = abs(ex - x) + abs(ey - y)
            cost = float(heu[y, x]) if heu is not None else 0.0
            eta = 1.0 / (1.0 + cost + dist)   # kleiner dist -> größere eta

            w = tau * (eta ** self.beta)
            if w <= 0:
                w = 1e-12
            weights.append(w)

        total = sum(weights)
        probs = [w / total for w in weights]

        # Roulette-Wheel-Auswahl
        r = self.rannumber()
        tmp = 0.0
        chosen_idx = len(neighbor_cords) - 1  # Fallback
        for i, p in enumerate(probs):
            tmp += p
            if r <= tmp:
                chosen_idx = i
                break

        chosen = neighbor_cords[chosen_idx]
        # Elterninfo für Pfadrekonstruktion speichern
        self.parents[chosen] = self.pos
        self.pos = chosen
        tx, ty = self.pos
        self.vis[ty, tx] = True

        # Ziel erreicht?
        if self.pos == self.end:
            self.goal = True
            return False

        return True  # Ameise kann in der nächsten Runde weiterlaufen

    def neighbor_dirs(self, x, y, maze, visited=False):
        h, w = maze.shape[:2]
        dirs, coords = [], []

        for k, (dx, dy, opp_idx, wall_idx) in DIR_MAP.items():
            nx, ny = x + dx, y + dy

            if not in_bounds(nx, ny, w, h):
                continue

            cell = maze[y, x]
            if cell[wall_idx]:
                continue

            if self.vis[ny, nx] != visited:
                continue

            dirs.append(k)
            coords.append((nx, ny))

        return dirs, coords

    def allowed_moves(self, maze):
        x, y = self.pos
        dirs_, neighbor_cords = self.neighbor_dirs(x, y, maze, visited=False)
        return dirs_, neighbor_cords

    def pathreconstruction(self):
        """
        Rekonstruiert den Pfad vom Start bis zur aktuellen Position self.pos.
        Funktioniert sowohl für erfolgreiche (goal) als auch erfolglose Ameisen.
        """
        self.path = [self.pos]
        # von aktueller Position zurücklaufen, solange es einen Parent gibt
        while self.path[-1] in self.parents:
            self.path.append(self.parents[self.path[-1]])
        self.path.reverse()  # jetzt: [Start, ..., aktuelle Position]

    def place_pheromone(self, pher):
        """
        Trägt Pheromon entlang des gelaufenen Pfades auf.
        Auch nicht-zielerreichende Ameisen verstärken ihre (Teil-)Wege.
        Dabei werden Pfade, deren Endpunkt näher am Ziel liegt, stärker gewichtet.
        """
        if not self.path:
            return

        last_x, last_y = self.path[-1]
        dist_goal =( abs(last_x - self.end[0])**2 + abs(last_y - self.end[1])**2)**(1/2)


        amount = (self.max_pheromone) / len(self.path) * (self.decayrate)
        for (x, y) in self.path:
            pher[y, x] += amount
