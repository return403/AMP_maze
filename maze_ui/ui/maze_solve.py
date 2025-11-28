
import pygame
import pygame_gui
from .base import AlgUI
from maze_algorithms import bfs, A_star
from maze_ants_colony import ants_colony
from maze_core import h_euk, h_man

class MazeSolve(AlgUI):
    options = ["BFS", "A*", "Ant"]
    DICT = {"BFS": bfs, "A*": A_star, "Ant":ants_colony}

    options_A_star = [ "h_man","h_euk"]
    DICT_A_star = {"h_euk": h_euk, "h_man": h_man}

    def __init__(self, x_pos, y_pos, manager, x_max=2000, y_max=2000, dx=10, dy=20, xs_size=200, x_size=60, y_size=30):
        """
        Initialisiert die MazeSolve-UI für die Auswahl und Konfiguration von Maze-Lösungsalgorithmen.
        
        Parameter:
            x_pos: X-Position der UI.
            y_pos: Y-Position der UI.
            manager: pygame_gui Manager.
            x_max, y_max: Maximale Koordinaten.
            dx, dy: Abstand zwischen UI-Elementen.
            xs_size: Breite des Dropdown-Menüs.
            x_size, y_size: Größe der Buttons.
        """
        super().__init__(self.options, x_pos, y_pos, manager)
        self.manager = manager
        self.layout = {"x": x_pos, "y": y_pos, "dx": dx, "dy": dy, "x_size": x_size, "y_size": y_size, "xs_size": xs_size}
        self.label__ = []
        self.label__.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos+ 13*(y_size +dy)-25,
                    xs_size, 
                    y_size
                ),text="Wähle Heuristik:", manager=self.manager))
        self.label__.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos+ 14*(y_size +dy)-25,
                    xs_size, 
                    y_size
                ),text="Weight, Kosten", manager=self.manager))
        
        self.heu_dropdown = pygame_gui.elements.UIDropDownMenu(options_list=self.options_A_star, starting_option=self.options_A_star[0], relative_rect=pygame.Rect(x_pos, y_pos+13*(y_size +dy), xs_size, y_size), manager=manager)
        self.weight_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_pos+0*(dx+x_size), y_pos+ 14*(y_size +dy)), (x_size, y_size)), manager=manager)
        self.weight_submit = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_pos + 2 * (x_size + dx), y_pos+ 14*(y_size +dy)), (x_size, y_size)), text="OK", manager=manager)
        self.weight_input.set_text("1.0")
        self.cost_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_pos+1*(dx+x_size), y_pos+ 14*(y_size +dy)), (x_size, y_size)), manager=manager)
        
        self.cost_input.set_text("0")
        self.selected_heuristic = self.options_A_star[0]
        self.data_ = (1.0, 0, self.DICT_A_star[self.selected_heuristic])
        self.hide_()
    
    def handle_event(self, event):
        """
        Verarbeitet UI-Events für die MazeSolve-Komponente.
        
        Parameter:
            event: Pygame-Event zum Verarbeiten.
        
        Rückgabe:
            Tuple mit Event-Informationen oder None.
        """
        result = super().handle_event(event)
        if result:
            _, typ, value = result
            if typ == "dropdown":
                if value == "A*":
                    self.show_()
                else:
                    self.hide_()
            return (typ, value, "c_solve")
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.weight_submit:
                try:
                    w = float(self.weight_input.get_text())
                    v = int(self.cost_input.get_text())
                except ValueError:
                    w = 1.0
                    v = 0
                w = max(0.0, w)
                v = max(0, v)
                self.data_ = (w, v, self.DICT_A_star[self.selected_heuristic])
                return "submit_", self.data_, "c_solve"
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.heu_dropdown:
            self.selected_heuristic = event.text
            w = self.data_[0] if self.data_ else 1.0
            v = self.data_[1] if self.data_ else 0
            self.data_ = (w, v, self.DICT_A_star[self.selected_heuristic])
    
    def show_(self):
        """
        Zeigt die Heuristik- und Gewichtungs-UI-Elemente an.
        """
        self.heu_dropdown.show()
        self.weight_input.show()
        self.weight_submit.show()
        self.cost_input.show()
        for i in self.label__:
            i.show()
    
    def hide_(self):
        """
        Versteckt die Heuristik- und Gewichtungs-UI-Elemente.
        """
        self.heu_dropdown.hide()
        self.weight_input.hide()
        self.weight_submit.hide()
        self.cost_input.hide()
        for i in self.label__:
            i.hide()
