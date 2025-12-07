
"""
Hauptprogramm für Maze-Generator und Solver.
Verwaltet die Hauptschleife, Event-Verarbeitung und Visualisierung von Algorithmenschritten.
"""
import time
import psutil
import os
import pygame
from collections import deque
from maze_core import initialize_maze, maze_to_img
from maze_graphics import draw_cell, init_pygame, Color, Game

process = psutil.Process(os.getpid())


# ===== Farbmapping für Algorithmus-Schritte =====
STEP_COLORS = {
    "start": Color.TEAL.value,
    "close": Color.GREEN.value,
    "open": Color.MAGENTA.value,
    "frontier": Color.ORANGE.value,
    "frontierNew": Color.ORANGE.value,
    "Path": Color.RED.value,
    "done_bfs": None,  # Keine Aktion
    "done": Color.TEAL.value,
    "randomWall": None,  # Hat "from" und "to" statt "pos"
}

STEP_TRANSITIONS = {
    "forward": (Color.ORANGE.value, Color.RED.value),
    "backtrack": (Color.TEAL.value, Color.ORANGE.value),
    "frontierNew": (Color.TEAL.value, Color.TEAL.value),
    "randomWall": (Color.TEAL.value, Color.TEAL.value),
}


def get_step_color(step: dict, color_scale) -> tuple:
    """Bestimmt die Anzeigefarbe für einen Algorithmus-Visualisierungsschritt.
    
    Mapt verschiedene Schritt-Typen auf RGB-Farbwerte. Berücksichtigt Sonderfälle
    wie BFS mit Distanz-Färbung.
    
    Args:
        step: Event-Dict vom Algorithmus-Generator mit "type" und optionalen Daten.
        color_scale: Farbskala für Distanz-basierte Färbung (z.B. BFS).
    
    Returns:
        RGB-Farbtuple (r, g, b) oder None wenn kein visueller Output nötig.
    """
    step_type = step.get("type")
    
    # Standard-Farben aus Mapping
    if step_type in STEP_COLORS:
        if step_type == "BFS":
            # Spezialfall: Distanz-basierte Färbung
            v = max(0, int(step.get("value", 0)))
            c = int(color_scale[v])
            return (0, c, c)
        return STEP_COLORS.get(step_type)
    
    return None


def handle_single_cell_step(screen, game, step: dict) -> None:
    """Visualisiert einen Algorithmenschritt, der eine einzelne Zelle färbt.
    
    Rendert Zelle mit entsprechender Farbe basierend auf Schritt-Typ.
    Aktualisiert nur die betroffene Zelle auf dem Screen.
    
    Args:
        screen: Pygame-Surface zum Zeichnen.
        game: Game-Objekt mit Maze, Dimensionen und Farbskala.
        step: Event-Dict mit "type" und "pos" (x, y) Koordinaten.
    """
    step_type = step.get("type")
    
    # Skip steps ohne Position
    if "pos" not in step:
        return
    
    # BFS hat Distanz-basierte Färbung
    if step_type == "BFS":
        v = max(0, int(step.get("value", 0)))
        c = int(game.color_scale[v])
        color = (0, c, c)
    else:
        color = get_step_color(step, game.color_scale)
        if color is None:
            return
    
    x, y = step["pos"]
    draw_cell(screen, game.maze, x, y, game.cell_size, color)
    pygame.display.update(pygame.Rect(
        x * game.cell_size, y * game.cell_size,
        game.cell_size, game.cell_size
    ))


def handle_transition_step(screen, game, step: dict) -> None:
    """Visualisiert Übergänge zwischen zwei Zellen (Forward, Backtrack, etc.).
    
    Rendert zwei Zellen mit Übergangsfarben basierend auf Bewegungstyp.
    Nutzt STEP_TRANSITIONS Dict für Farbzuordnung.
    
    Args:
        screen: Pygame-Surface zum Zeichnen.
        game: Game-Objekt mit Maze und Dimensionen.
        step: Event-Dict mit "type", "from" und "to" Koordinaten-Tupeln.
    """
    step_type = step.get("type")
    if step_type not in STEP_TRANSITIONS:
        return
    
    color_old, color_new = STEP_TRANSITIONS[step_type]
    old_x, old_y = step.get("from", (0, 0))
    new_x, new_y = step.get("to", (0, 0))
    
    draw_cell(screen, game.maze, old_x, old_y, game.cell_size, color_old)
    draw_cell(screen, game.maze, new_x, new_y, game.cell_size, color_new)
    
    pygame.display.update([
        pygame.Rect(old_x * game.cell_size, old_y * game.cell_size,
                    game.cell_size, game.cell_size),
        pygame.Rect(new_x * game.cell_size, new_y * game.cell_size,
                    game.cell_size, game.cell_size),
    ])


def process_algorithm_step(screen, game, step: dict) -> None:
    """Zentrale Verarbeitung von Algorithmus-Visualisierungsschritten.
    
    Routet verschiedene Schritt-Typen an spezialisierte Handler (einzelne Zelle vs. Übergänge).
    
    Args:
        screen: Pygame-Surface zum Zeichnen.
        game: Game-Objekt mit aktueller Maze und Dimensionen.
        step: Event-Dict vom Algorithmus-Generator.
    """
    step_type = step.get("type")
    
    # Einzelzell-Schritte
    if step_type in {"start", "frontier", "done", "done_bfs", "BFS", "Path", "close", "open"}:
        handle_single_cell_step(screen, game, step)
    
    # Übergänge
    elif step_type in {"forward", "backtrack", "frontierNew", "randomWall"}:
        handle_transition_step(screen, game, step)


def main():
    """Hauptprogrammschleife für Maze-Generator/Solver.
    
    Verwaltet Pygame-Fenster, Event-Loop, Algorithmus-Ausführung mit/ohne Animation,
    und Statistik-Ausgabe.
    """
    placeholder_maze = initialize_maze(5, 5)
    
    clock, screen, manager = init_pygame(placeholder_maze)
    running = True
    start_time = None
    
    sw, sh = screen.get_size()
    game = Game(0, 0, manager, screen)
    
    while running:
        # Event-Verarbeitung
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            manager.process_events(event)
            game.handle_event(event)
        
        # Speed vom Settings-UI auslesen
        speed = game.settings_ui.speed
        
        # Algorithmus-Schritte verarbeiten
        if game.algorithm_running and game.generator is not None:
            if start_time is None:
                start_time = time.perf_counter()
            
            if game.animation:
                # Mit Animation: Schritt für Schritt
                try:
                    step = next(game.generator)
                    process_algorithm_step(screen, game, step)
                
                except StopIteration:
                    finished_all = game.on_algorithm_finished()
                    if finished_all and start_time is not None:
                        end_time = time.perf_counter()
                        elapsed = end_time - start_time
                        print(f"Laufzeit: {elapsed:.6f} Sekunden")
                        print(game.menu_tab)
                        game.show_stats(len(game.solve_path), elapsed, game.menu_tab)
                        print(f"{process.memory_info().rss / (1024 ** 2):.4f} MB")
                        
                        # Exportiere Maze nur bei Generierungsalgorithmen
                        if game.algorithm_type == "generate":
                            maze_h, maze_w = game.maze.shape[:2]
                            maze_to_img(game.maze, filename=f"maze_{maze_h}x{maze_w}_{elapsed:.6f}.png")
                        elif game.algorithm_type == "solve":
                            maze_h, maze_w = game.maze.shape[:2]
                            maze_to_img(game.maze, filename=f"maze_{maze_h}x{maze_w}_{elapsed:.6f}.png", solve_path=game.solve_path)
                        
                        start_time = None
                
                except Exception as ex:
                    print(f"Fehler im Algorithmus: {ex}")
                    game.algorithm_running = False
            
            else:
                # Ohne Animation: Alle Schritte direkt durchlaufen
                deque(game.generator, maxlen=0)
                
                finished_all = game.on_algorithm_finished()
                if finished_all and start_time is not None:
                    end_time = time.perf_counter()
                    elapsed = end_time - start_time
                    print(f"Laufzeit: {elapsed:.6f} Sekunden")
                    print(game.menu_tab)
                    game.show_stats(len(game.solve_path), elapsed, game.menu_tab)
                    print(f"{process.memory_info().rss / (1024 ** 2):.4f} MB")
                    maze_h, maze_w = game.maze.shape[:2]
                    maze_to_img(game.maze, filename=f"maze_{maze_h}x{maze_w}_{elapsed:.6f}.png")

                    start_time = None
        
        # UI-Panel zeichnen
        ui_panel_rect = pygame.Rect(sw - 240, 0, 240, sh)
        pygame.draw.rect(screen, Color.BLACK.value, ui_panel_rect)
        
        # UI aktualisieren
        # Bei 10%: 5 FPS (0.2s pro Schritt), bei 100%: 300 FPS
        fps = 5 + (speed - 10) / 90 * 295
        manager.update(clock.tick(fps) / 1000.0)
        manager.draw_ui(screen)
        pygame.display.update()
        
        # Weitere Rendering-Operationen
        game.gen_ui.draw(screen)
        game.sol_ui.draw(screen)


if __name__ == "__main__":
    main()
