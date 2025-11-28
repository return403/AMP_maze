
"""Koordinaten-Eingabekomponente für UI."""
import pygame
import pygame_gui


class CoordInput:
    """
    Eingabekomponente für X,Y-Koordinaten mit optionalem Mouse-Capture.
    Bietet zwei Textfelder für X- und Y-Eingaben sowie einen Button für Mouse-Capture oder Bestätigung.
    
    Attribute:
        MOUSE_ACTIVE_COLOR: Farbe für aktiven Mouse-Capture
        MOUSE_INACTIVE_COLOR: Farbe für inaktiven Mouse-Capture
    """
    
    # Button-Status Farben
    MOUSE_ACTIVE_COLOR = pygame.Color("#880000")
    MOUSE_INACTIVE_COLOR = pygame.Color("#4c5052")
    
    def __init__(
        self,
        x_pos: int,
        y_pos: int,
        manager: pygame_gui.UIManager,
        x_set_text: str = "0",
        y_set_text: str = "0",
        x_size: int = 60,
        y_size: int = 30,
        dx: int = 10,
        catch_mode: bool = True,
        cell_size: int = 1,
    ):
        """
        Initialisiert die Koordinateneingabe-Komponente.
        Erstellt zwei Textfelder (X, Y) und einen Button (Mouse-Capture oder OK).
        
        Parameter:
            x_pos: X-Position des UI-Elements.
            y_pos: Y-Position des UI-Elements.
            manager: pygame_gui UIManager für die Verwaltung.
            x_set_text: Initialtext für das X-Textfeld.
            y_set_text: Initialtext für das Y-Textfeld.
            x_size: Breite der Textfelder.
            y_size: Höhe der Textfelder.
            dx: Horizontaler Abstand zwischen den Elementen.
            catch_mode: True=Mouse-Button, False=OK-Button.
            cell_size: Zellgröße für Mouse-Capture-Skalierung.
        """
        self.manager = manager
        self.cell_size = cell_size
        self.catch_mode = catch_mode
        self.mouse_catch = False
        
        # Eingabefelder
        self.x = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((x_pos, y_pos), (x_size, y_size)),
            manager=manager,
        )
        self.y = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((x_pos + x_size + dx, y_pos), (x_size, y_size)),
            manager=manager,
        )
        
        # Mouse-Button oder Submit-Button
        button_x = x_pos + 2 * (x_size + dx)
        if catch_mode:
            self.mouse = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect((button_x, y_pos), (x_size, y_size)),
                text="mouse",
                manager=manager,
            )
            self.submit = None
        else:
            self.submit = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect((button_x, y_pos), (x_size, y_size)),
                text="OK",
                manager=manager,
            )
            self.mouse = None
        
        # Initialtexte setzen
        self.x.set_text(x_set_text)
        self.y.set_text(y_set_text)
    
    def handle_event(self, event: pygame.event.EventType):
        """
        Verarbeitet UI-Events (Button, Mausklicks, Texteingabe).
        
        Parameter:
            event: Pygame-Event zum Verarbeiten.
        
        Rückgabe:
            Tuple (action_type, data) oder None:
            - ("toggle", bool): Mouse-Capture-Status getoggelt.
            - ("set", (x, y)): Mauskoordinate erfasst.
            - ("ok", (x_text, y_text)): Submit-Button gedrückt.
        """
        # Mouse-Button Toggle
        if self.mouse is not None and event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.mouse:
                self.mouse_catch = not self.mouse_catch
                self._update_mouse_color()
                return "toggle", self.mouse_catch
        
        # Mouse-Klick im Catch-Modus
        if self.catch_mode and self.mouse_catch:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                mx //= self.cell_size
                my //= self.cell_size
                self.x.set_text(str(mx + 1))
                self.y.set_text(str(my + 1))
                self.mouse_catch = False
                self._update_mouse_color()
                return "set", (mx, my)
        
        # Submit-Button
        if self.submit is not None and event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.submit:
                return "ok", (self.x.get_text(), self.y.get_text())
        
        return None
    
    def _update_mouse_color(self) -> None:
        """
        Aktualisiert die Farbe des Mouse-Buttons basierend auf dem Aktivierungsstatus.
        """
        if self.mouse is None:
            return
        
        color = self.MOUSE_ACTIVE_COLOR if self.mouse_catch else self.MOUSE_INACTIVE_COLOR
        self.mouse.colours.update({
            "normal_bg": color,
            "hovered_bg": color,
        })
        self.mouse.rebuild()
    
    def update_cell_size(self, cell_size: int) -> None:
        """
        Aktualisiert die Zellgröße für die Maus-Koordinaten-Skalierung.
        
        Parameter:
            cell_size: Neue Zellgröße in Pixeln.
        """
        self.cell_size = cell_size
    
    def _iter_elements(self):
        """
        Iteriert über alle UI-Elemente dieser Komponente.
        
        Gibt zurück:
            pygame_gui UI-Elemente.
        """
        yield self.x
        yield self.y
        if self.mouse is not None:
            yield self.mouse
        if self.submit is not None:
            yield self.submit
    
    def hide(self) -> None:
        """
        Versteckt alle UI-Elemente dieser Komponente.
        """
        for element in self._iter_elements():
            element.hide()
    
    def show(self) -> None:
        """
        Zeigt alle UI-Elemente dieser Komponente.
        """
        for element in self._iter_elements():
            element.show()
    
    def kill(self) -> None:
        """
        Löscht alle UI-Elemente.
        """
        for element in self._iter_elements():
            element.kill()
