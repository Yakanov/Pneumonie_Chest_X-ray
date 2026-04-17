@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ================================================================
REM  Lance l'API Flask du projet.
REM  Si l'environnement virtuel n'existe pas encore, on appelle d'abord
REM  le script d'installation.
REM ================================================================

if not exist ".venv\Scripts\python.exe" (
    echo Environnement Python non detecte. Installation en cours...
    call setup_project.bat --no-pause
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Lancement de l'API Flask...
python api_flask_pneumonia.py

if errorlevel 1 (
    echo.
    echo [ERREUR] L'API s'est arretee avec une erreur.
    pause
    exit /b 1
)
