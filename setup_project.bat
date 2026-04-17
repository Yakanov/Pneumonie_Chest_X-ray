@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "NO_PAUSE="
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

REM ================================================================
REM  Script d'installation automatique du projet
REM  Objectif :
REM  1) Detecter Python
REM  2) Creer un environnement virtuel local
REM  3) Installer toutes les dependances
REM  4) Laisser le projet pret pour l'API et le notebook
REM ================================================================

echo.
echo [1/4] Detection d'une installation Python compatible...
set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.13 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.13"
    if not defined PYTHON_CMD py -3.12 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.12"
    if not defined PYTHON_CMD py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
    if not defined PYTHON_CMD py -3.10 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.10"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERREUR] Python n'a pas ete trouve sur cette machine.
    echo Installez Python puis relancez ce script.
    pause
    exit /b 1
)

echo Python detecte : %PYTHON_CMD%

echo.
echo [2/4] Creation de l'environnement virtuel local...
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERREUR] Echec lors de la creation du venv.
        pause
        exit /b 1
    )
) else (
    echo Environnement .venv deja present.
)

echo.
echo [3/4] Mise a jour de pip et installation des dependances...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERREUR] Echec lors de la mise a jour de pip.
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERREUR] Echec lors de l'installation des dependances.
    pause
    exit /b 1
)

echo.
echo [4/4] Verification rapide de l'installation...
python -c "import flask, tensorflow, sklearn, pandas, seaborn, imblearn, PIL, joblib; print('Verification OK')"
if errorlevel 1 (
    echo [ERREUR] Les dependances ne semblent pas correctement installees.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Installation terminee avec succes.
echo.
echo Dataset :
echo - Si le dataset existe deja dans un dossier standard du projet, le notebook le trouvera automatiquement.
echo - Sinon, le notebook tentera de le telecharger depuis Kaggle.
echo - Si Kaggle n'est pas configure, ajoutez %%USERPROFILE%%\.kaggle\kaggle.json
ECHO   ou definissez KAGGLE_USERNAME et KAGGLE_KEY.
echo.
echo Scripts disponibles :
echo - setup_project.bat   : installation de l'environnement
echo - run_api.bat         : lancement de l'API Flask
echo - run_notebook.bat    : ouverture du notebook dans Jupyter Lab
echo ================================================================
echo.
if not defined NO_PAUSE pause
