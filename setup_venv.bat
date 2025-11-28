@echo off
setlocal

set "VENV=.venv"
set "REQ=requirements.txt"
set "VENV_PY=%VENV%\Scripts\python.exe"

rem Python-Launcher wählen
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

echo [1/6] Virtuelle Umgebung erstellen
%PY% -m venv "%VENV%" || goto :err

echo [2/6] Pip aktualisieren
"%VENV_PY%" -m pip install --upgrade pip || goto :err

echo [3/6] Altes pygame entfernen (falls vorhanden)
"%VENV_PY%" -m pip uninstall -y pygame >nul 2>nul

echo [4/6] requirements.txt anlegen, falls nicht vorhanden
if not exist "%REQ%" (
  (
    echo numpy
    echo psutil
    echo Pillow
    echo pygame-ce^>=2.5.5,^<3.0
    echo pygame-gui==0.6.14
    echo matplotlib
  ) > "%REQ%"
)

echo [5/6] Abhaengigkeiten installieren
"%VENV_PY%" -m pip install -r "%REQ%" || goto :err

echo [6/6] Kurztest der Imports
"%VENV_PY%" -c "import pygame, pygame_gui, numpy, psutil, PIL; print('ok')" || goto :err

if exist "main.py" (
  echo Starte main.py ...
  "%VENV_PY%" main.py
) else (
  echo Fertig. Starte dein Programm spaeter mit:
  echo "%VENV_PY%" main.py
)

echo Done.
exit /b 0

:err
echo Fehler. Code %errorlevel%.
exit /b %errorlevel%
