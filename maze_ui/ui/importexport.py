# -*- coding: utf-8 -*-
"""
Import/Export-UI für Maze und Heatmaps.
Verwaltet Dateiauswahl, Import und Export von generierten Labyrinthen.
"""
from typing import List, Dict, Tuple, Optional
import pygame
import pygame_gui
from pathlib import Path
import json
import numpy as np
from maze_core import import_img


class importexport:
    """
    UI-Komponente für Maze- und Heatmap-Import/Export.
    Bietet Dateiauswahl und Gewichtungs-Kontrolle für Importe und Exporte.
    
    Attribute:
        OPTIONS: Verfügbare Import/Export-Typen
        FILE_PATTERNS: Dateimuster für die Auswahl
        DEFAULT_WEIGHT: Standardgewichtung
        MIN_WEIGHT: Minimale Gewichtung
    """
    
    # ===== Klassenkonstanten =====
    OPTIONS = ["Maze", "Map"]
    FILE_PATTERNS = {"Maze": "*.json", "Map": "*.jpg"}
    DEFAULT_WEIGHT = 1.0
    MIN_WEIGHT = 1.0
    
    def __init__(self, x_pos, y_pos, manager, x_max=2000, y_max=2000, dx=10, dy=20, xs_size=200, x_size=60, y_size=30):
        
        self.layout = {"x": x_pos, "y": y_pos, "dx": dx, "dy": dy, "x_size": x_size, "y_size": y_size, "xs_size": xs_size}
        self.manager = manager
        self.options_files = ["_placeHolder"]
        self.folder_path = Path(__file__).resolve().parents[2]
        self.data_ = self.OPTIONS[0]
        self.file_name = "_placeHolder"
        self.label = []
        
        self.label.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos -25,
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Wähle Maze oder Heatmaps:", manager=manager))
        
        self.label.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos -25 + 1*(y_size +dy),
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Wähle Datei:", manager=manager))
        
        self.label.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos -25 + 2*(y_size +dy),
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Import / Export", manager=manager))
        
        self.dropdown = pygame_gui.elements.UIDropDownMenu(options_list=self.OPTIONS, starting_option=self.data_, relative_rect=pygame.Rect(x_pos, y_pos+0*(y_size +dy), xs_size, y_size), manager=manager)
        
        self.options_files_dropdown = pygame_gui.elements.UIDropDownMenu(options_list=self.options_files, starting_option=self.options_files[0], relative_rect=pygame.Rect(x_pos, y_pos+1*(y_size +dy), xs_size, y_size), manager=manager)
        self.refresh_list(self.data_)
        self.file_import = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_pos + 0 * (x_size + dx), y_pos+ 2*(y_size +dy)), (x_size, y_size)), text="Import", manager=manager)
        self.file_export = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_pos + 2 * (x_size + dx), y_pos+ 2*(y_size +dy)), (x_size, y_size)), text="Export", manager=manager)
        self.file_name = self.options_files[0]
        self.label_ = []
        
        
        self.label_.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos -25 + 3*(y_size +dy),
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Edit Heatmap", manager=manager))
        self.label_.append(pygame_gui.elements.UILabel(relative_rect=pygame.Rect(
                    x_pos,                      
                    y_pos -25 + 4*(y_size +dy),
                    self.layout["xs_size"], 
                    self.layout["y_size"]
                ),text="Heatmap Weight", manager=manager))
        self.show_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.layout["x"] + 1 * (self.layout["x_size"] + self.layout["dx"]), self.layout["y"]+ 3*(self.layout["y_size"] +self.layout["dy"])), (self.layout["x_size"], self.layout["y_size"])), text="Show", manager=manager)
        self.clear_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.layout["x"] + 0 * (self.layout["x_size"] + self.layout["dx"]), self.layout["y"]+ 3*(self.layout["y_size"] +self.layout["dy"])), (self.layout["x_size"], self.layout["y_size"])), text="Clear", manager=manager)
        self.add_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.layout["x"] + 2 * (self.layout["x_size"] + self.layout["dx"]), self.layout["y"]+ 3*(self.layout["y_size"] +self.layout["dy"])), (self.layout["x_size"], self.layout["y_size"])), text="Add", manager=manager)
        self.mapWeight = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((self.layout["x"] + 0 * (self.layout["x_size"] + self.layout["dx"]), self.layout["y"]+ 4*(self.layout["y_size"] +self.layout["dy"])), (self.layout["xs_size"], self.layout["y_size"])), manager=manager)
        self.hide()
        self.weight = 1.0
        self.mapWeight.set_text(str(self.weight))
        
    def refresh_list(self, mode):
        """
        Aktualisiert die Dateiliste im Dropdown-Menü basierend auf dem Modus.
        
        Parameter:
            mode: Modus ("Maze" oder "Map")
        """
        if self.options_files_dropdown:
            self.options_files_dropdown.kill()

        self.options_files = []
        pattern = self.FILE_PATTERNS.get(mode, "*.json")
        for p in self.folder_path.glob(pattern):
            self.options_files.append(p.name)
        if not self.options_files:
            self.options_files.append("_placeHolder")
        self.options_files_dropdown = pygame_gui.elements.UIDropDownMenu(options_list=self.options_files, starting_option=self.options_files[0], 
                                                                         relative_rect=pygame.Rect(self.layout["x"], self.layout["y"]+1*(self.layout["y_size"] +self.layout["dy"]), self.layout["xs_size"], self.layout["y_size"]), manager=self.manager)
        
    def import_grid(self, dateiname):
        """
        Importiert ein Maze-Grid aus einer JSON-Datei.
        
        Parameter:
            dateiname: Name der zu importierenden Datei.
        
        Rückgabe:
            arr: Numpy-Array mit Maze-Daten.
        """
        p = self.folder_path / dateiname
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        arr = np.asarray(d["grid"], dtype=bool)
        assert arr.shape == (d["h"], d["w"], 5)
        return arr
                
    def export_grid(self, arr):
        """
        Exportiert ein Maze-Grid als JSON-Datei.
        
        Parameter:
            arr: Numpy-Array mit Maze-Daten.
        
        Rückgabe:
            name: Dateiname der exportierten Datei.
        """
        h, w = arr.shape[:2]
        assert arr.ndim == 3 and arr.shape[2] == 5
        name = f"{w}_{h}_maze.json"
        p = self.folder_path / name
        with p.open("w", encoding="utf-8") as o:
            json.dump({"w": w, "h": h, "grid": arr.tolist()}, o)
        return name
    
    def import_map(self, maze, weight, img_path):
        """
        Importiert eine Map/Heatmap aus einem Bild.
        
        Parameter:
            maze: Maze-Daten.
            weight: Gewichtung.
            img_path: Pfad zum Bild.
        
        Rückgabe:
            Ergebnis von import_img.
        """
        return import_img(maze, weight, path=str(self.folder_path / img_path))
    
    def handle_event(self, event):
        """
        Verarbeitet UI-Events für Import/Export und Map-Funktionen.
        
        Parameter:
            event: Pygame-Event zum Verarbeiten.
        
        Rückgabe:
            Tuple mit Event-Informationen oder None.
        """
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.options_files_dropdown:
                self.file_name = event.text
                return ("dropdown_files", self.file_name, "c_IO")
            if event.ui_element == self.dropdown:
                self.data_ = event.text
                self.refresh_list(event.text)
                self.file_name = self.options_files[0]
                if self.data_ == "Map":
                    self.show_map_buttons()
                else:
                    self.hide_map_buttons()
                return ("dropdown", self.data_, "c_IO")
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.file_export:
                return ("submit_export", None, "c_IO")
            if event.ui_element == self.file_import:
                try:
                    w = float(self.mapWeight.get_text())
                except ValueError:
                    w = 1.0
                self.weight = max(1.0, w)
                return ("submit_import", self.weight, "c_IO")
            if event.ui_element == self.add_button:
                try:
                    w = float(self.mapWeight.get_text())
                except ValueError:
                    w = 1.0
                self.weight = max(1.0, w)
                return ("add", self.weight, "c_IO")
            if event.ui_element == self.clear_button:
                return ("clear", None, "c_IO")
            if event.ui_element == self.show_button:
                return ("show", None, "c_IO")
    
    def hide_map_buttons(self):
        self.clear_button.hide()
        self.add_button.hide()
        self.show_button.hide()
        self.mapWeight.hide()
        for i in self.label_:
            i.hide()
    
    def show_map_buttons(self):
        self.clear_button.show()
        self.add_button.show()
        self.show_button.show()
        self.mapWeight.show()
        for i in self.label_:
            i.show()
    
    def hide(self):
        self.dropdown.hide()
        self.options_files_dropdown.hide()
        self.file_import.hide()
        self.file_export.hide()
        self.hide_map_buttons()
        for i in self.label:
            i.hide()
        
    def show(self):
        self.dropdown.show()
        self.options_files_dropdown.show()
        self.file_import.show()
        self.file_export.show()
        for i in self.label:
            i.show()