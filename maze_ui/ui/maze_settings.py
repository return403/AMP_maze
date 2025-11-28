"""Settings-Panel für Animation Control."""
import pygame
import pygame_gui


class Settings:
    """
    Einfaches Settings-Panel mit Animation-Toggle.
    
    Attribute:
        ENABLED_COLOR: Farbe für aktivierte Animation
        DISABLED_COLOR: Farbe für deaktivierte Animation
    """
    
    # Farben
    ENABLED_COLOR = pygame.Color("#008843")
    DISABLED_COLOR = pygame.Color("#4c5052")
    
    def __init__(self, x_pos: int, y_pos: int, manager: pygame_gui.UIManager):
        """
        Initialisiert das Settings-Panel.
        
        Parameter:
            x_pos: X-Position der UI-Elemente.
            y_pos: Y-Position der UI-Elemente.
            manager: pygame_gui Manager.
        """
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_size = 170
        self.y_size = 30
        
        # Animation-Status
        self.animation_enabled = True
        
        # Label
        self.label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (x_pos, y_pos - 25),
                (self.x_size, self.y_size),
            ),
            text="Toggle Animation:",
            manager=manager,
        )
        
        # Animation-Toggle Button
        self.animation_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_pos, y_pos),
                (self.x_size, self.y_size),
            ),
            text=self._get_button_text(),
            manager=manager,
        )
        self._update_button_appearance()
    
    def _get_button_text(self) -> str:
        """
        Generiert den Button-Text basierend auf dem Animation-Status.
        """
        return f"Animation: {'ON' if self.animation_enabled else 'OFF'}"
    
    def _update_button_appearance(self) -> None:
        """
        Aktualisiert das Button-Erscheinungsbild und den Text.
        """
        color = self.ENABLED_COLOR if self.animation_enabled else self.DISABLED_COLOR
        self.animation_button.colours.update({
            "normal_bg": color,
            "hovered_bg": color,
        })
        self.animation_button.set_text(self._get_button_text())
        self.animation_button.rebuild()
    
    def handle_event(self, event: pygame.event.EventType):
        """
        Verarbeitet Button-Klicks.
        
        Rückgabe:
            Tuple ("animation", bool, "Settings") oder None.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.animation_button:
                self.animation_enabled = not self.animation_enabled
                self._update_button_appearance()
                return ("animation", self.animation_enabled, "Settings")
        
        return None
    
    def hide(self) -> None:
        """
        Versteckt alle UI-Elemente.
        """
        self.animation_button.hide()
        self.label.hide()
    
    def show(self) -> None:
        """
        Zeigt alle UI-Elemente.
        """
        self.animation_button.show()
        self.label.show()
