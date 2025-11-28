
import pygame
import pygame_gui

class Analyze:
    """
    UI-Komponente zur Analyse und Vermessung von Generierungsalgorithmen.
    Erlaubt die Eingabe von Parametern für die Messung und die Ausführung.
    
    Attribute:
        layout: Dictionary mit Layout-Parametern
        x_max, y_max: Maximale Koordinaten
        coord_list: Liste der Koordinaten-Eingaben
    """
    
    def __init__(self, x_pos, y_pos, manager, x_max=2000, y_max=2000, dx=10, dy=20, xs_size=200, x_size=60, y_size=30, n_coords=2):
        """
        Initialisiert die Analyze-UI.
        
        Parameter:
            x_pos: X-Position der UI.
            y_pos: Y-Position der UI.
            manager: pygame_gui Manager.
            x_max, y_max: Maximale Koordinaten.
            dx, dy: Abstand zwischen UI-Elementen.
            xs_size: Breite des Labels.
            x_size, y_size: Größe der Buttons.
            n_coords: Anzahl der Koordinateneingaben.
        """
        self.manager = manager
        self.layout = {"x": x_pos, "y": y_pos, "dx": dx, "dy": dy, "x_size": x_size, "y_size": y_size, "xs_size": xs_size, "n_coords": n_coords}
        self.x_max, self.y_max = x_max - 1, y_max - 1
        self.bg_color = (0, 0, 0)
        self.coord_list = []
        self.label = []
        
        self.label.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos -25,
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Gen-Algorithmen Vermessen", manager=manager))
        
        self.label.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos+ y_size +dy -25,
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Max-Size/Steps/Repeats", manager=manager))
        
        self.start = self._make_button("start", 0, 0)
        self.cancel = self._make_button("cancel", 2, 0)
        self.submit = self._make_button("submit", 1, 0)

        self.max_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_pos+0*(dx+x_size), y_pos+ y_size +dy), (x_size, y_size)), manager=manager)
        self.step_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_pos+1*(dx+x_size), y_pos+ y_size +dy), (x_size, y_size)), manager=manager)
        self.repeat_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_pos+2*(dx+x_size),y_pos + y_size +dy), (x_size, y_size)), manager=manager)

    def hide(self):
        """
        Versteckt alle UI-Elemente.
        """
        for el in (self.start, self.cancel, self.submit, self.max_input, self.step_input, self.repeat_input):
            if el is not None:
                el.hide()
        for i in self.label:
            i.hide()

    def show(self):
        """
        Zeigt alle UI-Elemente.
        """
        for el in (self.start, self.cancel, self.submit, self.max_input, self.step_input, self.repeat_input):
            if el is not None:
                el.show()
        for i in self.label:
            i.show()

    def _make_button(self, text, col, row, width=None):
        """
        Erstellt einen Button an einer bestimmten Gitter-Position.
        
        Parameter:
            text: Text des Buttons.
            col: Spalte im Gitter.
            row: Zeile im Gitter.
            width: Optionale Breite (Standard: x_size).
        
        Rückgabe:
            Der erstellte pygame_gui UIButton.
        """
        x = self.layout["x"] + col * (self.layout["x_size"] + self.layout["dx"])
        y = self.layout["y"] + row * (self.layout["y_size"] + self.layout["dy"])
        w = width if width else self.layout["x_size"]
        rect = pygame.Rect((x, y), (w, self.layout["y_size"]))
        return pygame_gui.elements.UIButton(relative_rect=rect, text=text, manager=self.manager)

    def handle_event(self, event):
        """
        Verarbeitet UI-Events für die Analyze-Komponente.
        
        Parameter:
            event: Pygame-Event zum Verarbeiten.
        
        Rückgabe:
            Tuple mit Event-Informationen oder None.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.start:
                return ("start", None, "c_analyze")
            if event.ui_element == self.cancel:
                return ("cancel", None, "c_analyze")
            if event.ui_element == self.submit:
                return ("submit", [self.max_input.get_text(),self.step_input.get_text(), self.repeat_input.get_text()], "c_analyze")
