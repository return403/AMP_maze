"""
Basis-UI Komponente für Algorithmus-Auswahl und Parametersteuerung.
Abstrakte Klasse mit gemeinsamen Funktionalitäten für Maze-Generation und -Lösung.
"""
from typing import List, Dict, Tuple, Optional
import pygame
import pygame_gui
from .coord_input import CoordInput


class AlgUI:
    """
    Basis-UI-Klasse für Algorithmen mit Koordinateneingabe und Dropdown-Auswahl.
    Verwaltet Buttons, Labels und Koordinaten-Inputs für die Konfiguration von Algorithmen.
    
    Attribute:
        DEFAULT_BG_COLOR: Standard-Hintergrundfarbe
        MAX_WAYPOINTS: Maximale Anzahl an Zwischenstopps
        MIN_WAYPOINTS: Minimale Anzahl an Zwischenstopps
    """
    
    # ===== Klassenkonstanten =====
    DEFAULT_BG_COLOR = (0, 0, 0)
    MAX_WAYPOINTS = 10
    MIN_WAYPOINTS = 2
    
    def __init__(
        self, 
        options: List[str], 
        x_pos: int, 
        y_pos: int, 
        manager: pygame_gui.UIManager,
        x_max: int = 2000, 
        y_max: int = 2000, 
        dx: int = 10, 
        dy: int = 20, 
        xs_size: int = 200, 
        x_size: int = 60, 
        y_size: int = 30, 
        n_coords: int = 2, 
        allow_resize: bool = True
    ):
        """
        Initialisiert die Algorithmus-UI.
        
        Parameter:
            options: Liste der verfügbaren Algorithmen.
            x_pos, y_pos: Position der UI-Elemente.
            manager: pygame_gui Manager.
            x_max, y_max: Maximale Koordinaten (Weltgrenzen).
            dx, dy: Abstand zwischen UI-Elementen.
            xs_size: Breite des Dropdown-Menüs.
            x_size, y_size: Größe der Buttons.
            n_coords: Anzahl der Koordinateneingaben.
            allow_resize: True, wenn die Größe angepasst werden kann.
        """
        self.manager = manager
        self.allow_resize = allow_resize
        self.layout: Dict[str, int] = {
            "x": x_pos, "y": y_pos, 
            "dx": dx, "dy": dy, 
            "x_size": x_size, "y_size": y_size, 
            "xs_size": xs_size, "n_coords": n_coords
        }
        self.x_max, self.y_max = x_max - 1, y_max - 1
        self.bg_color = self.DEFAULT_BG_COLOR
        self.coord_list: List[CoordInput] = []
        self.cell_size = 1
        
        # Dropdown für Algorithmusauswahl
        self.dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=options, 
            starting_option=options[0], 
            relative_rect=pygame.Rect(x_pos, y_pos, xs_size, y_size),
            manager=manager
        )
        self.selected_alg = options[0]
        
        # Standard-Buttons
        self.start = self._make_button("start", 0, 1)
        self.cancel = self._make_button("cancel", 2, 1)
        self.submit = self._make_button("submit", 1, 1)
        
        # Optional: Größen-Adjust-Buttons
        self.add = self._make_button("+", 1, 2) if self.allow_resize else None
        self.sub = self._make_button("-", 2, 2) if self.allow_resize else None
        
        # Labels für Überschriften und Koordinaten
        self.labels: List[pygame_gui.elements.UILabel] = []
        self.label_: List[pygame_gui.elements.UILabel] = []
        
        self._create_static_labels(x_pos, y_pos, xs_size, y_size)
        self._initialize_coordinates(x_pos, y_pos, dy, y_size, n_coords)
    
    def _create_static_labels(self, x_pos: int, y_pos: int, xs_size: int, y_size: int) -> None:
        """
        Erstellt statische Labels (Überschriften) der UI.
        
        Parameter:
            x_pos: X-Position.
            y_pos: Y-Position.
            xs_size: Breite des Labels.
            y_size: Höhe des Labels.
        """
        self.labels.append(
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(x_pos, y_pos - 25, xs_size, y_size),
                text="Wähle Algorithmus:",
                manager=self.manager
            )
        )
        self.labels.append(
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(x_pos, y_pos + y_size - 5, xs_size, y_size),
                text="Steuerung",
                manager=self.manager
            )
        )
        if self.allow_resize:
            self.labels.append(
                pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect(x_pos, y_pos + y_size + 70-25, xs_size, y_size),
                    text="+/- Zwischenstopps",
                    manager=self.manager
                )
            )
    
    def _initialize_coordinates(self, x_pos: int, y_pos: int, dy: int, y_size: int, n_coords: int) -> None:
        """
        Erstellt die initialen Koordinateneingaben und die zugehörigen Labels.
        
        Parameter:
            x_pos: X-Position.
            y_pos: Y-Position.
            dy: Vertikaler Abstand.
            y_size: Höhe der Eingabefelder.
            n_coords: Anzahl der Koordinatenfelder.
        """
        for i in range(n_coords):
            label_text = self._get_coord_label_text(i, n_coords)
            self.label_.append(
                pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect(
                        x_pos,
                        y_pos - 25 + (i + 2) * (y_size + dy),
                        self.layout["xs_size"],
                        y_size
                    ),
                    text=label_text,
                    manager=self.manager
                )
            )
            self._add_coord_input()
    
    def _get_coord_label_text(self, index: int, total: int) -> str:
        """
        Gibt den passenden Label-Text für eine Koordinateneingabe zurück.
        
        Parameter:
            index: Index des Koordinatenfeldes (0-basiert).
            total: Gesamtzahl der Koordinatenfelder.
        
        Rückgabe:
            Beschreibungstext für das Label (z.B. "Startpunkt (x,y)").
        """
        if index == 0:
            return "Größe (n,m)" if not self.allow_resize else "Startpunkt (x,y)"
        elif index == total - 1:
            return "Startpunkt (x,y)" if not self.allow_resize else "Endpunkt (x,y)"
        else:
            return f"{index}. Zwischenstopp (x,y)"
    
    def _make_button(self, text: str, col: int, row: int, width: Optional[int] = None) -> pygame_gui.elements.UIButton:
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
        w = width or self.layout["x_size"]
        rect = pygame.Rect((x, y), (w, self.layout["y_size"]))
        return pygame_gui.elements.UIButton(relative_rect=rect, text=text, manager=self.manager)
    
    def _add_coord_input(self, cell_size: int = 1) -> CoordInput:
        """
        Fügt eine neue Koordinateneingabe hinzu (maximal 10 Zwischenstopps).
        Erstellt einen neuen CoordInput mit Mouse-Capture-Funktion und aktualisiert die Labels.
        
        Parameter:
            cell_size: Zellgröße für Mouse-Capture-Skalierung.
        
        Rückgabe:
            Die erstellte oder letzte CoordInput-Instanz.
        """
        if len(self.coord_list) >= self.MAX_WAYPOINTS:
            return self.coord_list[-1]
        
        # Berechne Y-Position
        temp_offset = -1 if not self.allow_resize else 0
        y_offset = (len(self.coord_list) + 3 + temp_offset) * (self.layout["y_size"] + self.layout["dy"])
        
        # Erstelle Input
        coord_input = CoordInput(
            self.layout["x"],
            self.layout["y"] + y_offset,
            self.manager,
            x_size=self.layout["x_size"],
            y_size=self.layout["y_size"],
            dx=self.layout["dx"],
            cell_size=cell_size
        )
        self.coord_list.append(coord_input)
        self._update_coord_labels()
        return coord_input
    
    def _remove_coord_input(self) -> None:
        """
        Entfernt den letzten Coordinate-Input, behält aber mindestens MIN_WAYPOINTS.
        """
        if len(self.coord_list) <= self.MIN_WAYPOINTS:
            return
        
        coord_input = self.coord_list.pop()
        coord_input.kill()
        self._update_coord_labels()
    
    def _update_coord_labels(self) -> None:
        """
        Aktualisiert alle Labels für Coordinate-Inputs nach Änderungen.
        Löscht alte Labels und erstellt neue basierend auf aktueller Inputanzahl.
        """
        # Lösche alte Labels
        for label in self.label_:
            label.kill()
        self.label_.clear()
        
        # Erstelle neue Labels
        temp_offset = -1 if not self.allow_resize else 0
        for i in range(len(self.coord_list)):
            label_text = self._get_coord_label_text(i, len(self.coord_list))
            self.label_.append(
                pygame_gui.elements.UILabel(
                    relative_rect=pygame.Rect(
                        self.layout["x"],
                        self.layout["y"] - 25 + (i + 3 + temp_offset) * (self.layout["y_size"] + self.layout["dy"]),
                        self.layout["xs_size"],
                        self.layout["y_size"]
                    ),
                    text=label_text,
                    manager=self.manager
                )
            )
    
    def handle_event(self, event: pygame.event.Event) -> Optional[Tuple]:
        """
        Verarbeitet UI-Events (Buttons, Inputs, Dropdowns).
        Routet Events an CoordInputs, Buttons und Dropdown-Menü. Koordiniert Mouse-Capture zwischen mehreren CoordInputs.
        
        Parameter:
            event: Pygame-Event zum Verarbeiten.
        
        Rückgabe:
            Optionales Tuple (self, event_type, data) oder None, wenn nicht relevant.
        """
        # Verarbeite CoordInput-Events
        for coord_input in self.coord_list:
            result = coord_input.handle_event(event)
            if result:
                kind, data = result
                if kind == "set":
                    return (self, "coord_set", (coord_input, data))
                if kind == "toggle" and data:
                    # Deaktiviere andere Inputs
                    for other in self.coord_list:
                        if other is not coord_input:
                            other.mouse_catch = False
                            other._update_mouse_color()
                    return (self, "coord_toggle", coord_input)
        
        # Verarbeite Button-Clicks
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.start:
                return (self, "start", None)
            if event.ui_element == self.cancel:
                return (self, "cancel", None)
            if event.ui_element == self.submit:
                return (self, "submit", self.get_points())
            if self.add is not None and event.ui_element == self.add:
                self._add_coord_input(cell_size=self.cell_size)
            if self.sub is not None and event.ui_element == self.sub:
                self._remove_coord_input()
        
        # Verarbeite Dropdown
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.dropdown:
            self.selected_alg = event.text
            return (self, "dropdown", event.text)
        
        return None
    
    def get_points(self) -> List[Tuple[int, int]]:
        """
        Liest alle Koordinateneingaben aus und validiert sie.
        Konvertiert die Texteingaben zu Integern und begrenzt sie auf die Weltgrenzen (x_max, y_max).
        Gibt eine leere Liste zurück bei Parsing-Fehlern.
        
        Rückgabe:
            Liste von (x, y) Koordinaten-Tupeln innerhalb der Weltgrenzen.
        """
        points: List[Tuple[int, int]] = []
        for coord_input in self.coord_list:
            try:
                x = min(self.x_max, max(0, int(coord_input.x.get_text()))) - 1
                y = min(self.y_max, max(0, int(coord_input.y.get_text()))) - 1
                points.append((x, y))
            except ValueError:
                print("Falsche Eingabe in Koordinatenfeld")
        return points
    
    def hide(self) -> None:
        """
        Versteckt alle UI-Elemente dieser Komponente.
        """
        for element in [self.dropdown, self.start, self.cancel, self.submit, self.add, self.sub]:
            if element is not None:
                element.hide()
        
        for coord_input in self.coord_list:
            coord_input.hide()
        for label in self.labels + self.label_:
            label.hide()
    
    def show(self) -> None:
        """
        Zeigt alle UI-Elemente dieser Komponente.
        """
        for element in [self.dropdown, self.start, self.cancel, self.submit, self.add, self.sub]:
            if element is not None:
                element.show()
        
        for coord_input in self.coord_list:
            coord_input.show()
        for label in self.labels + self.label_:
            label.show()
    
    def draw(self, screen: pygame.Surface) -> None:
        """
        Zeichnet den Panel-Hintergrund und die Komponenten.
        
        Parameter:
            screen: Pygame-Surface zum Zeichnen.
        """
        panel_height = (len(self.coord_list) + 4) * (self.layout["y_size"] + self.layout["dy"])
        rect = pygame.Rect(
            self.layout["x"],
            self.layout["y"],
            self.layout["xs_size"],
            panel_height
        )
        pygame.draw.rect(screen, self.bg_color, rect)
        self.manager.draw_ui(screen)
