
"""Hauptmenü-Navigationskomponente."""
import pygame
import pygame_gui


class MenuSelection:
    """
    Hauptmenü mit Navigation zwischen verschiedenen Modi.
    
    Attribute:
        ACTIVE_COLOR: Farbe für aktiven Button
        INACTIVE_COLOR: Farbe für inaktiven Button
        TABS: Liste der verfügbaren Tabs
    """
    
    # Farben für Button-Status
    ACTIVE_COLOR = pygame.Color("#008843")
    INACTIVE_COLOR = pygame.Color("#4c5052")
    
    # Verfügbare Menu-Tabs
    TABS = ["Generate", "Solve", "Settings", "Image", "Import/Export", "Analyze"]
    
    def __init__(self, x_pos: int, y_pos: int, manager: pygame_gui.UIManager):
        """
        Initialisiert das Hauptmenü.
        
        Parameter:
            x_pos: X-Position des ersten Buttons.
            y_pos: Y-Position des ersten Buttons.
            manager: pygame_gui Manager.
        """
        self.button_pos_x = x_pos
        self.button_pos_y = y_pos
        self.manager = manager
        self.x_size = 170
        self.y_size = 30
        self.active_button = None
        
        # Erstelle Buttons für jeden Tab
        self.buttons = {}
        for idx, tab_name in enumerate(self.TABS):
            rect = pygame.Rect(
                (self.button_pos_x + idx * self.x_size, self.button_pos_y),
                (self.x_size, self.y_size),
            )
            self.buttons[tab_name] = pygame_gui.elements.UIButton(
                relative_rect=rect,
                text=tab_name,
                manager=manager,
            )
        
        # Exit-Button
        self.exit = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (self.button_pos_x + len(self.TABS) * self.x_size, self.button_pos_y),
                (self.x_size, self.y_size),
            ),
            text="Exit",
            manager=manager,
        )
        
        # Initialisiere mit Generate als aktiv
        self.set_active(self.buttons["Generate"])
    
    def set_active(self, active_button: pygame_gui.elements.UIButton) -> None:
        """
        Markiert einen Button als aktiv (farblich hervorgehoben).
        
        Parameter:
            active_button: Der zu aktivierende Button.
        """
        for button in self.buttons.values():
            color = self.ACTIVE_COLOR if button == active_button else self.INACTIVE_COLOR
            button.colours.update({"normal_bg": color, "hovered_bg": color})
            button.rebuild()
        self.active_button = active_button
    
    def handle_event(self, event: pygame.event.EventType):
        """
        Verarbeitet Button-Klicks.
        
        Rückgabe:
            Tuple (self, tab_name, "menu") oder None.
        """
        if event.type != pygame_gui.UI_BUTTON_PRESSED:
            return None
        
        # Tab-Buttons
        for tab_name, button in self.buttons.items():
            if event.ui_element == button:
                self.set_active(button)
                return (self, tab_name, "menu")
        
        # Exit-Button
        if event.ui_element == self.exit:
            pygame.quit()
            return None
        
        return None
