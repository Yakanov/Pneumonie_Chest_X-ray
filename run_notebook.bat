@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ================================================================
REM  Ouvre le notebook principal dans Jupyter Lab.
REM  Le notebook lui-meme gere automatiquement la recherche du dataset
REM  et tente un telechargement Kaggle si necessaire.
REM ================================================================

if not exist ".venv\Scripts\python.exe" (
    echo Environnement Python non detecte. Installation en cours...
    call setup_project.bat --no-pause
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Ouverture du notebook principal...
python -m jupyter lab "Projet Science des donnees\Classification_CNN_Pneumonia_OPTIMIZED.ipynb"

if errorlevel 1 (
    echo.
    echo [ERREUR] Impossible d'ouvrir Jupyter Lab.
    pause
    exit /b 1
)
