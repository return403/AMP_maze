# AMP Maze (Python)

Ein interaktives Maze-Generator- und Solver-Tool mit visueller Darstellung (Pygame + pygame_gui).

Kurzbeschreibung
-----------------
AMP Maze ist ein kleines Visualisierungsprojekt, das verschiedene Labyrinth-Generierungs- und Lösungsalgorithmen implementiert und deren Ablauf schrittweise darstellt. Es eignet sich zum Lernen, Experimentieren mit Heuristiken und zum Benchmarken der Algorithmen.

Die Kernidee ist, die Algorithmen als Generatoren zu implementieren, die Ereignisse (z. B. "forward", "backtrack", "done") yielden. Diese Ereignisse werden in der GUI visualisiert; im "measure_mode" laufen Algorithmen ohne UI-Events für saubere Laufzeitmessungen.

Ausführlichere Features
-----------------------
- Generator-basierte Implementierung: Jeder Algorithmus liefert Ereignisse, daher sind Visualisierung, Pause/Resume und Benchmarking einfach möglich.
- Generatoren/Algorithmen:
  - DFS (Depth-First Search) Generator mit Backtracking-Visualisierung
  - Randomized Prim's Algorithmus für Frontier-basierte Generierung
  - BFS Solver mit Distanz-Gradienten-Visualisierung
  - A* Solver mit wählbaren Heuristiken (Manhattan, Euklidisch) und optionaler Heatmap-Gewichtung
- Heatmap-Unterstützung: Import/Export von Heatmaps (Bild-basierte Gewichtung) und Anzeige als Graustufen-Overlay (geht nur für A*)
- GUI:
  - Steuerbare Animation (step-by-step) oder schneller Messmodus
  - Eingabefelder für Maze-Größe, Start-/Zielpunkte und Algorithmusoptionen
  - Anzeige von Laufzeit und Pfadlänge
- Benchmarks: Skript/funktion für automatisierte Benchmarks mehrerer Algorithmen über verschiedene Maze-Größen

Voraussetzungen
--------------
- Python 3.10+ (getestet mit 3.14)
- Siehe `requirements.txt` für Abhängigkeiten (`pygame`, `pygame_gui`, `numpy`, `psutil`, `Pillow`)

Schnellstart (Windows / PowerShell)
---------------------------------
1. Virtuelle Umgebung erstellen und aktivieren:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Abhängigkeiten installieren:

```powershell
pip install -r requirements.txt
```

3. Anwendung starten:

```powershell
# GUI starten
python main.py
# oder per Skript
.\start.bat
```

Projektstruktur (wichtigste Dateien)
----------------------------------
- `main.py`  Hauptschleife, Event-Loop und Renderer-Glue
- `maze_core.py`  Kernfunktionen, Datenstrukturen und Heuristiken
- `maze_algorithms.py`  Implementierung der Generatoren/Solver (yield-basiert)
- `maze_ui/`  GUI-Module (UI-Komponenten, Runtime, Render)
- `requirements.txt`  Python-Abhängigkeiten


Lizenz
------
Dieses Projekt steht unter der MIT-Lizenz.


