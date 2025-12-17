
import pygame
import pygame_gui
import numpy as np
from collections import deque
from maze_algorithms import benchmark, benchmark_sol
from maze_core import initialize_maze, reset_visited, Open_Random_Walls, export_img, maze_to_img
from ..ui.menu import MenuSelection
from ..ui.maze_gen import MazeGen
from ..ui.maze_solve import MazeSolve
from ..ui.analyze import Analyze
from ..ui.importexport import importexport
from ..ui.maze_settings import Settings
from ..render import draw_cell
from ..colors import Color

class Game:
    """
    handelt alle Events der Unterklassen.
    speichert alle wichtigen Variablen in sich.
    """
    
    def __init__(self, x_pos, y_pos, manager, screen):
        self.manager = manager
        self.layout = {"x": x_pos, "y": y_pos, "dx": 10, "dy": 20, "x_size": 60, "y_size": 30, "xs_size": 200}
        self.screen = screen
        self.width, self.height = self.screen.get_size()
        self.draw_box = (self.width-250, self.height-50)
        
        # UI-Elemente initialisieren
        ui_x_pos = self.width - 200
        self.menu_ui = MenuSelection(0, self.height-30, manager)
        self.gen_ui = MazeGen(ui_x_pos, y_pos+20, manager)
        self.sol_ui = MazeSolve(ui_x_pos, y_pos+20, manager)
        self.analyze_ui = Analyze(ui_x_pos, y_pos+20, manager)
        self.settings_ui = Settings(ui_x_pos, y_pos+20, manager)
        self.importexport_ui = importexport(ui_x_pos, y_pos+20, manager)
        
        # Algorithmus-Status
        self.generator = None
        self.algorithm_running = False
        self.algorithm_type = None  # 'generate' oder 'solve'
        self.animation = True
        
        # Rendering und Geometrie
        self.cell_size = 1
        self.capture_rect = None
        self.cached_surface = None
        self._geom_dirty = True
        self.color_scale = 0
        
        # Maze und Pathfinding
        self.maze = initialize_maze(10, 10, [False, False, False, False, False])
        self.solve_points = None
        self.solve_path = []
        self.solve_segment_index = 0
        
        # Heatmaps und UI
        self.heat_map = []
        self.label_stats = []
        self.menu_tab = " "
        self.data_gen = None  # Separate data for gen benchmark
        self.data_sol = None  # Separate data for sol benchmark
        
        # Initialisierungsschritte
        self.hide_all()
        self._setup_initial_ui()
        self.update_render_geometry()

    def _setup_initial_ui(self):
        """
        Initialisiert die Standard-UI-Einstellungen beim Start.
        """
        self.gen_ui.show()
        self.gen_ui.coord_list[0].x.set_text("10")
        self.gen_ui.coord_list[0].y.set_text("10")
        self.sol_ui.coord_list[-1].x.set_text("10")
        self.sol_ui.coord_list[-1].y.set_text("10")
        self.gen_ui.show_()
        # Ensure the menu tab state matches the visible UI so stats display works
        self.menu_tab = "Generate"
    
    def show_stats(self, length, time, menu):
        """
        Zeigt je nach Menu unterschiedlich die Algrotihmus Daten an.

        Parameters
        ----------
        length : int
            Pfadlänge.
        time : float
            Alg Dauer.
        menu : string
            Alg modus.

        Returns
        -------
        None.

        """
        
        
        if self.label_stats:
            for i in self.label_stats:
                i.kill()
                
        temp = 1 if menu == "Solve" else 0
        if menu == "Solve":
            self.label_stats.append(
                pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect(
                        self.width-250,
                        self.layout["y"] + (17)*(self.layout["y_size"] +self.layout["dy"]),
                        self.layout["xs_size"],
                        self.layout["y_size"]
                    ),
                    text=f"Pfad Länge: {length}",
                    manager=self.manager
                )
            )
        if menu in {"Solve", "Generate"}:
            self.label_stats.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                        self.width-250,
                        self.layout["y"]-25 + (17+temp)*(self.layout["y_size"] +self.layout["dy"]),
                        self.layout["xs_size"],
                        self.layout["y_size"]
                    ),
                    text=f"Dauer: {time:.4f} s", manager=self.manager))
    
    def hide_all(self):
        """
        Versteckt alle UI-Elemente

        Returns
        -------
        None.

        """
        self.gen_ui.hide()
        self.gen_ui.hide_()
        self.sol_ui.hide()
        self.sol_ui.hide_()
        self.analyze_ui.hide()
        self.importexport_ui.hide()
        self.settings_ui.hide()
        if self.label_stats:
            for i in self.label_stats:
                i.hide()
                  
    def update_render_geometry(self):
        """
        updated bei neue erstelltem oder importiertem Maze die Renderdaten wie Cell_size

        Returns
        -------
        None.

        """
        if self.maze is None:
            return
        mh, mw = self.maze.shape[:2]
        mw = max(1, mw); mh = max(1, mh)
        
        self.cell_size = max(1, min(self.draw_box[0] // mw, (self.draw_box[1]) // mh))
        self.sol_ui.cell_size = self.cell_size
        
        for i in self.gen_ui.coord_list:
            i.update_cell_size(self.cell_size)
        for i in self.sol_ui.coord_list:
            i.update_cell_size(self.cell_size)
        self.capture_rect = pygame.Rect(0, 0, mw * self.cell_size + 1, mh * self.cell_size + 1)
        self._geom_dirty = False

    def cache_from_screen(self, screen):
        """
        speichert die Fläche screen in self.cached_surface

        Parameters
        ----------
        screen : TYPE
            Die zu bemalene Fläche.

        Returns
        -------
        None.

        """
        if self.capture_rect and self.capture_rect.width > 0 and self.capture_rect.height > 0:
            self.cached_surface = pygame.Surface(self.capture_rect.size)
            self.cached_surface.blit(screen, (0, 0), self.capture_rect)

    def screen_from_cache(self, screen):
        """
        bemalt die Fläche screen mit dem im self.cached_surface gespeicherten bild.

        Parameters
        ----------
        screen : TYPE
            Die zu bemalene Fläche.

        Returns
        -------
        None.

        """
        if self.cached_surface is not None:
            screen.blit(self.cached_surface, (self.capture_rect.x, self.capture_rect.y))
            pygame.display.update(self.capture_rect)

    def clear_screen(self, screen, full = False):
        """
        Malt alles schwarz im gegebenen bereich

        Parameters
        ----------
        screen : TYPE
            Die zu bemalene Fläche.
        full : Bool, optional
            Entscheidet ob ganze Drawbox oder nur ein teil bemalt wird. The default is False.

        Returns
        -------
        None.

        """
        if self.capture_rect is not None:
            
            if not full:
                pygame.draw.rect(screen, (0, 0, 0), self.capture_rect)
            else:
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0,0,self.draw_box[0],self.draw_box[1]))
    
    def draw_full_maze(self, screen):
        """
        Zeichnet das Ganze Maze ohne animation

        Parameters
        ----------
        screen : TYPE
            Die zu bemalene Fläche.

        Returns
        -------
        None.

        """
        if self.maze is None:
            return

        h, w = self.maze.shape[:2]
        for y in range(h):
            for x in range(w):
                draw_cell(screen, self.maze, x, y, self.cell_size,Color.TEAL.value)
    
    def start_solve_segment(self):
        """
        Startet das aktuelle Segment (solve_segment_index) zwischen zwei Punkten.
        Wird sowohl beim ersten Start als auch nach jedem fertigen Segment aufgerufen.
        """
        if self.solve_points is None or self.maze is None:
            return
        if self.solve_segment_index >= len(self.solve_points) - 1:
            if self.screen is not None and self.solve_path:
                self.screen_from_cache(self.screen)
                for x, y in self.solve_path:
                    draw_cell(self.screen, self.maze, x, y, self.cell_size, color=Color.RED.value)
                pygame.display.update()
    
            self.algorithm_running = False
            self.generator = None
            self.solve_points = None
            self.solve_segment_index = 0
            return
    
        (sx, sy) = self.solve_points[self.solve_segment_index]
        (ex, ey) = self.solve_points[self.solve_segment_index + 1]
    
        h, w = self.maze.shape[:2]
        sx, sy = max(0, min(w - 1, sx)), max(0, min(h - 1, sy))
        ex, ey = max(0, min(w - 1, ex)), max(0, min(h - 1, ey))
    
        if self.screen is not None:
            self.screen_from_cache(self.screen)
    
        reset_visited(self.maze)
        
        alg_name = self.sol_ui.selected_alg
        if alg_name == "BFS":
            self.generator = self.sol_ui.DICT[alg_name](
                sx, sy, self.maze, ex, ey,
                False,           
                self.solve_path 
            )
    
        elif alg_name == "A*":
            if self.sol_ui.data_:
                weight, costs, heu = self.sol_ui.data_
            else:
                weight, costs, heu = 1.0, 0, None
    
            self.generator = self.sol_ui.DICT[alg_name](
                sx, sy, self.maze, ex, ey,
                heu,
                weight,
                self.sum_heatmaps(),
                False,            
                self.solve_path  
            )
        elif alg_name == "Ant":
            self.generator = self.sol_ui.DICT[alg_name](
                sx,sy, self.maze,ex,ey,
                heat_maps = self.sum_heatmaps(),
                path_list = self.solve_path)
        else:
            print("Unbekannter Solver:", alg_name)
            self.generator = None
            return
    
        self.algorithm_running = self.generator is not None

    def on_algorithm_finished(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        """
        Wird von main.py aufgerufen, wenn ein Generator StopIteration geworfen hat.
        Kümmert sich um Multi-Segment-Solving und sagt zurück,
        ob wirklich alles fertig ist (True) oder das nächste Segment gestartet wurde (False).
        """
        if self.solve_points is not None:
            self.solve_segment_index += 1
            self.start_solve_segment()
            return self.solve_points is None
        else:
            if self.screen is not None:
                if not self.animation:
                    self.draw_full_maze(self.screen)
                    pygame.display.update()
                
                self.cache_from_screen(self.screen)
            
            self.algorithm_running = False
            self.generator = None
            return True

    def sum_heatmaps(self):
        """
        Addiert alle Heatmaps zusammen

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Falls es gar keine Heatmaps gibt -> 0-Matrix in Maze-Größe
        if self.maze is None:
            return None  # oder np.array([]), je nach dem, was dein Solver erwartet
    
        h, w = self.maze.shape[:2]
    
        # Noch nichts geladen
        if self.heat_map is None or (isinstance(self.heat_map, (list, tuple)) and len(self.heat_map) == 0):
            return np.zeros((h, w), dtype=float)
    
        # Bereits eine einzige 2D-Map
        if isinstance(self.heat_map, np.ndarray):
            if self.heat_map.shape == (h, w):
                return self.heat_map
            else:
                # Falsche Form -> auf 0-Map zurückfallen oder Fehler werfen
                return np.zeros((h, w), dtype=float)
    
        # Liste von 2D-Maps -> summieren
        return np.sum(self.heat_map, axis=0)
    
    def _switch_menu_tab(self, tab_name):
        """
        Wechselt zwischen verschiedenen Menu-Tabs (Generate, Solve, Settings, etc.)
        
        Parameters
        ----------
        tab_name : str
            Name des Tabs: "Generate", "Solve", "Settings", "Image", "Import/Export", "Analyze"
        """
        self.menu_tab = tab_name
        self.hide_all()
        self.data_gen = None
        self.data_sol = None
        
        tab_handlers = {
            "Generate": lambda: (self.gen_ui.show_(), self.gen_ui.show()),
            "Solve": lambda: (self.sol_ui.show(), self.sol_ui.show_() if self.sol_ui.selected_alg == "A*" else None),
            "Settings": lambda: self.settings_ui.show(),
            "Image": lambda: None,
            "Import/Export": lambda: (self.importexport_ui.show(), 
                                      self.importexport_ui.refresh_list(self.importexport_ui.data_),
                                      self.importexport_ui.show_map_buttons() if self.importexport_ui.data_ == "Map" else None),
            "Analyze": lambda: self.analyze_ui.show(),
        }
        
        if tab_name in tab_handlers:
            tab_handlers[tab_name]()

    def _cancel_algorithm(self):
        """Bricht den aktuell laufenden Algorithmus ab."""
        self.algorithm_running = False
        self.generator = None
        self.solve_points = None
        self.solve_segment_index = 0
        self.solve_path = []

    def _start_generation(self, points):
        """
        Startet die Maze-Generierung mit den gegebenen Parametern.
        
        Parameters
        ----------
        points : list
            Liste der Koordinaten [(width, height), (start_x, start_y)]
        """
        if self.screen is not None and self.capture_rect is not None:
            self.clear_screen(self.screen)
        
        # Lösche alte Solve-Daten beim Start einer neuen Generierung
        self.solve_path = []
        self.solve_points = None
        self.solve_segment_index = 0
        
        maze_w = maze_h = 10
        start_x = start_y = 0
        if len(points) >= 1:
            maze_w, maze_h = max(2, points[0][0]+1), max(2, points[0][1]+1)
        if len(points) >= 2:
            start_x, start_y = points[1]

        self.maze = initialize_maze(maze_w, maze_h)
        self.color_scale = np.linspace(55, 255, self.maze.size + 1, dtype=int)
        self._geom_dirty = True
        self.update_render_geometry()

        # Make sure menu_tab reflects that we are in Generate mode
        self.menu_tab = "Generate"

        start_x = max(0, min(maze_w - 1, start_x))
        start_y = max(0, min(maze_h - 1, start_y))
        alg_name = self.gen_ui.selected_alg

        if alg_name == "DFS":
            self.generator = self.gen_ui.DICT[alg_name](start_x, start_y, self.maze)
        elif alg_name == "Prim":
            self.generator = self.gen_ui.DICT[alg_name](start_x, start_y, self.maze)
        elif alg_name == "Init_empty":
            self.maze = initialize_maze(maze_w, maze_h, [False,False,False,False,False])
            for y in range(maze_h):
                for x in range(maze_w):
                    draw_cell(self.screen, self.maze, x, y, self.cell_size, color=Color.TEAL.value)
            self.cache_from_screen(self.screen)
            #export_img(self.cached_surface, filename=f"maze_{maze_h}x{maze_w}_{alg_name}.png")
            hm = self.sum_heatmaps()
            hm_to_use = hm if (hm is not None and hm.max() > 0) else None
            maze_to_img(self.maze, filename=f"maze_{maze_h}x{maze_w}_{alg_name}.png", heatmap=hm_to_use)
        else:
            print("Unbekannter Generator:", alg_name)
            self.generator = None
        
        self.algorithm_running = self.generator is not None
        self.algorithm_type = 'solve' if self.algorithm_running else None
        self.algorithm_type = 'generate' if self.algorithm_running else None

    def _start_solving(self, points):
        """
        Startet das Maze-Solving mit den gegebenen Punkten.
        
        Parameters
        ----------
        points : list
            Liste der Punkte [(start_x, start_y), (end_x, end_y), ...]
        """
        if len(points) < 2:
            print("Bitte mindestens Start- und Zielkoordinate eingeben (zwei Punkte).")
            return
        
        if self.maze is None:
            print("Kein Labyrinth vorhanden. Bitte zuerst generieren.")
            return
        
        self.clear_screen(self.screen, full=True)
        self.screen_from_cache(self.screen)
        
        self.solve_points = points
        self.solve_path = []
        self.solve_segment_index = 0
        self.start_solve_segment()

    def _handle_heatmap_operations(self, typ, mode):
        """
        Behandelt alle Heatmap-Operationen (Import, Export, Clear, Show, Add).
        
        Parameters
        ----------
        typ : str
            Operationstyp: "submit_import", "submit_export", "clear", "show", "add"
        mode : str
            Modus muss "c_IO" sein
        """
        if mode != "c_IO" or self.algorithm_running:
            return
        
        if typ == "submit_import" and self.importexport_ui.data_ == "Map":
            hm = self.importexport_ui.import_map(
                self.maze,
                self.importexport_ui.weight,
                self.importexport_ui.file_name
            )
            if hm is not None:
                self.heat_map = [hm]
        
        elif typ == "clear":
            self.heat_map = np.zeros(self.maze.shape[:2])
            h, w = self.maze.shape[:2]
            for y in range(h):
                for x in range(w):
                    draw_cell(self.screen, self.maze, x, y, self.cell_size, color=Color.BLACK.value)
            self.cache_from_screen(self.screen)
        
        elif typ == "add":
            hm = self.importexport_ui.import_map(
                self.maze,
                self.importexport_ui.weight,
                self.importexport_ui.file_name
            )
            if hm is None:
                return
            if self.heat_map is None or isinstance(self.heat_map, np.ndarray):
                self.heat_map = [hm]
            else:
                self.heat_map.append(hm)
        
        elif typ == "show":
            self._display_heatmaps()

    def _display_heatmaps(self):
        """Zeigt die summierten Heatmaps farblich auf dem Screen an."""
        if self.heat_map is None:
            return
        
        maps = [self.heat_map] if isinstance(self.heat_map, np.ndarray) else self.heat_map
        if not maps:
            return
        
        h, w = self.maze.shape[:2]
        for m_ in maps:
            if m_.shape != (h, w):
                return
        
        tmp = np.sum(maps, axis=0).astype(float)
        m = tmp.max()
        if m <= 0:
            m = 1.0
        
        norm = (tmp / m * 255).astype(np.uint8)
        
        for y in range(h):
            for x in range(w):
                g = int(norm[y, x])
                tmp_color = (g, g, g)
                draw_cell(self.screen, self.maze, x, y, self.cell_size, color=tmp_color)
        
        self.cache_from_screen(self.screen)

    def _handle_maze_io(self, typ, mode):
        """
        Behandelt Maze-Import und Export Operationen.
        
        Parameters
        ----------
        typ : str
            Operationstyp: "submit_import", "submit_export"
        mode : str
            Modus muss "c_IO" sein
        """
        if mode != "c_IO" or self.algorithm_running or self.maze is None:
            return
        
        if typ == "submit_export" and self.importexport_ui.data_ == "Maze":
            self.importexport_ui.export_grid(self.maze)
            self.importexport_ui.refresh_list(self.importexport_ui.data_)
        
        elif typ == "submit_import" and self.importexport_ui.data_ == "Maze":
            try:
                self.maze = self.importexport_ui.import_grid(self.importexport_ui.file_name)
                self.update_render_geometry()
                self.color_scale = np.linspace(55, 255, self.maze.size + 1, dtype=int)
                h, w = self.maze.shape[:2]
                self.clear_screen(self.screen, full=True)
                
                for y in range(h):
                    for x in range(w):
                        draw_cell(self.screen, self.maze, x, y, self.cell_size, color=Color.TEAL.value)
                self.cache_from_screen(self.screen)
                self.importexport_ui.refresh_list(self.importexport_ui.data_)
            except Exception:
                print("Files not Found")

    def handle_event(self, event):
        """
        Zentrale Event-Verarbeitung für alle UI-Komponenten und Algorithmen.

        Parameters
        ----------
        event : pygame.event.EventType
            Das zu verarbeitende Event
        """
        result = (self.gen_ui.handle_event(event) or 
                  self.sol_ui.handle_event(event) or 
                  self.analyze_ui.handle_event(event) or 
                  self.menu_ui.handle_event(event) or 
                  self.importexport_ui.handle_event(event) or 
                  self.settings_ui.handle_event(event))
        
        if not result:
            return
        
        typ, value, mode = result
        print(f"{mode}:{typ}:{value}")

        # Menu-Tab Wechsel
        menu_tabs = {"Settings", "Image", "Generate", "Solve", "Import/Export", "Analyze"}
        if isinstance(value, str) and value in menu_tabs:
            self._switch_menu_tab(value)
            return

        # Allgemeine Controls
        if typ == "cancel":
            self._cancel_algorithm()
            return
        
        if typ == "animation":
            self.animation = value
            return
        
        if typ == "submit" and mode == "c_gen":
            self.data_gen = [value, mode]
            return
        
        if typ == "submit" and mode == "c_solve":
            self.data_sol = [value, mode]
            return
        
        if typ == "submit" and mode == "c_analyze":
            # Check if it's submit (generation) or submit_sol (solver benchmark)
            # For now, store both - they'll be differentiated by the "start" vs "start_sol" handler
            self.data_gen = [value, mode]
            self.data_sol = [value, mode]
            return
        
        if typ == "submit_sol" and mode == "c_analyze":
            # Solver benchmark (no walls_list needed - auto-calculated)
            self.data_sol = [value, mode]
            return

        # Analyze-Modus
        if typ == "start" and mode == "c_analyze" and self.data_gen is not None:
            size_max, steps, repeats = self.data_gen[0]
            algs = [self.gen_ui.DICT[k] for k in self.gen_ui.options]
            algs.pop()  # drop Init_empty
            benchmark(int(size_max), int(steps), int(repeats), self.maze, algs)
            return
        
        if typ == "start_sol" and mode == "c_analyze" and self.data_sol is not None:
            size_max, steps, repeats = self.data_sol[0]
            # benchmark_sol berechnet Wandanzahl automatisch basierend auf Größe
            benchmark_sol(int(size_max), int(steps), int(repeats))
            return
        
        # Open Random Walls
        if typ == "submit_" and mode == "c_gen" and not self.algorithm_running:
            self.algorithm_running = True
            if self.screen is not None:
                self.screen_from_cache(self.screen)
            self.generator = Open_Random_Walls(self.maze, self.gen_ui.data_)
            return
        
        # Generation starten
        if typ == "start" and mode == "c_gen" and self.data_gen is not None:
            self.algorithm_type = "generate"
            self._start_generation(self.data_gen[0])
            return

        # Solving starten
        if typ == "start" and mode == "c_solve" and self.data_sol is not None:
            self.algorithm_type = "solve"
            self._start_solving(self.data_sol[0])
            return

        # Maze IO
        if typ in {"submit_export", "submit_import"} and mode == "c_IO":
            self._handle_maze_io(typ, mode)
            return

        # Heatmap IO
        if typ in {"submit_import", "clear", "add", "show"} and mode == "c_IO":
            self._handle_heatmap_operations(typ, mode)
