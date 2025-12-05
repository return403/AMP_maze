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
        self.speed = 50  # 0-100%
        
        # Label für Animation
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
        
        # Speed-Label
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (x_pos, y_pos + 50),
                (self.x_size, self.y_size),
            ),
            text="Speed: 50%",
            manager=manager,
        )
        
        # Speed-Slider (10-100)
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                (x_pos, y_pos + 85),
                (self.x_size, 25),
            ),
            start_value=50,
            value_range=(10, 100),
            manager=manager,
        )
    
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
        Verarbeitet Button-Klicks und Slider-Änderungen.
        
        Rückgabe:
            Tuple ("animation", bool, "Settings") für Animation,
            Tuple ("speed", int, "Settings") für Speed-Slider, oder None.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.animation_button:
                self.animation_enabled = not self.animation_enabled
                self._update_button_appearance()
                return ("animation", self.animation_enabled, "Settings")
        
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.speed_slider:
                self.speed = int(event.value)
                self.speed_label.set_text(f"Speed: {self.speed}%")
                return ("speed", self.speed, "Settings")
        
        return None
    
    def hide(self) -> None:
        """
        Versteckt alle UI-Elemente.
        """
        self.animation_button.hide()
        self.label.hide()
        self.speed_label.hide()
        self.speed_slider.hide()
    
    def show(self) -> None:
        """
        Zeigt alle UI-Elemente.
        """
        self.animation_button.show()
        self.label.show()
        self.speed_label.show()
        self.speed_slider.show()
