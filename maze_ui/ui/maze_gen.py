
import pygame_gui
import pygame
from .base import AlgUI
from maze_algorithms import dfs, prim

from ..render import init_empty

class MazeGen(AlgUI):
    options = ["DFS", "Prim", "Init_empty"]
    DICT = {"DFS": dfs, "Prim": prim, "Init_empty": init_empty}

    def __init__(self, x_pos, y_pos, manager: pygame_gui.UIManager, x_max=2000, y_max=2000, dx=10, dy=20, xs_size=200, x_size=60, y_size=30):
        """
        Initialisiert die MazeGen-UI für die Auswahl und Konfiguration von Maze-Generierungsalgorithmen.
        
        Parameter:
            x_pos: X-Position der UI.
            y_pos: Y-Position der UI.
            manager: pygame_gui Manager.
            x_max, y_max: Maximale Koordinaten.
            dx, dy: Abstand zwischen UI-Elementen.
            xs_size: Breite des Dropdown-Menüs.
            x_size, y_size: Größe der Buttons.
        """
        super().__init__(self.options, x_pos, y_pos, manager, allow_resize=False)
        self.manager = manager
        self.openwalls_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_pos+0*(dx+x_size), y_pos+ 5*(y_size +dy)), (x_size, y_size)), manager=manager)
        self.openwalls_submit = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_pos + 2 * (x_size + dx), y_pos+ 5*(y_size +dy)), (x_size, y_size)), text="OK", manager=manager)
        self.label = []
        
        self.label_.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos+ 5*(y_size +dy)-25,
                    xs_size, 
                    y_size
                ),text="Opens n- Walls randomly", manager=self.manager))
        
        self.data_ = 0
        
    def handle_event(self, event):
        """
        Verarbeitet UI-Events für die MazeGen-Komponente.
        
        Parameter:
            event: Pygame-Event zum Verarbeiten.
        
        Rückgabe:
            Tuple mit Event-Informationen oder None.
        """
        result = super().handle_event(event)
        if result:
            _, typ, value = result
            return (typ, value, "c_gen")
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.openwalls_submit:
                try:
                    w = max(0,int(self.openwalls_input.get_text()))
                except ValueError:
                    w = 0
                self.data_ = w
                return ("submit_",w,"c_gen")
                
    def show_(self):
        """
        Zeigt die Eingabefelder für offene Wände an.
        """
        self.openwalls_input.show()
        self.openwalls_submit.show()

    def hide_(self):
        """
        Versteckt die Eingabefelder für offene Wände.
        """
        self.openwalls_input.hide()
        self.openwalls_submit.hide()