#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script d'installation automatique du projet (Linux)
# Compatible en priorité avec Zorin OS, Ubuntu, Linux Mint, Pop!_OS
# et Debian. Peut aussi fonctionner sur d'autres distributions si
# Python, venv et les roues Python requises sont disponibles.
# ================================================================

cd "$(dirname "$0")"

detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "${PRETTY_NAME:-Linux}"
    else
        echo "Linux (distribution non identifiee)"
    fi
}

detect_support_level() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        local distro_id="${ID:-}"
        local distro_like=" ${ID_LIKE:-} "

        case "$distro_id" in
            zorin|ubuntu|linuxmint|pop|debian)
                echo "supportee"
                return
                ;;
            fedora)
                echo "probable"
                return
                ;;
            arch|manjaro)
                echo "probable"
                return
                ;;
        esac

        if [[ "$distro_like" == *" ubuntu "* ]] || [[ "$distro_like" == *" debian "* ]]; then
            echo "supportee"
        elif [[ "$distro_like" == *" fedora "* ]] || [[ "$distro_like" == *" arch "* ]]; then
            echo "probable"
        else
            echo "non_testee"
        fi
    else
        echo "non_testee"
    fi
}

print_support_banner() {
    local level="$1"
    echo
    echo "Diagnostic compatibilite Linux :"
    case "$level" in
        supportee)
            echo "- Statut : distribution supportee"
            echo "- Niveau de confiance : eleve"
            ;;
        probable)
            echo "- Statut : support probable"
            echo "- Niveau de confiance : moyen"
            echo "- Remarque : une petite adaptation systeme peut etre necessaire"
            ;;
        *)
            echo "- Statut : distribution non testee"
            echo "- Niveau de confiance : prudent"
            echo "- Remarque : le code peut fonctionner, mais l'environnement n'est pas garanti"
            ;;
    esac
}

print_venv_help() {
    echo
    echo "[AIDE] Le module venv n'est pas disponible dans Python." 
    echo "Installez-le selon votre distribution, puis relancez ce script."
    echo
    echo "Exemples :"
    echo "- Ubuntu / Zorin / Linux Mint / Pop!_OS : sudo apt install python3-venv"
    echo "- Debian : sudo apt install python3-venv"
    echo "- Fedora : sudo dnf install python3-virtualenv"
    echo "- Arch / Manjaro : sudo pacman -S python-virtualenv"
}

DISTRO_NAME="$(detect_distro)"
SUPPORT_LEVEL="$(detect_support_level)"

echo
echo "Distribution detectee : $DISTRO_NAME"
print_support_banner "$SUPPORT_LEVEL"
echo "[1/5] Detection d'une installation Python compatible..."

PYTHON_CMD=""
for candidate in python3.11 python3.10 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_CMD="$candidate"
        break
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "[ERREUR] Python n'a pas ete trouve sur cette machine."
    echo "Installez Python 3 puis relancez ce script."
    exit 1
fi

PYTHON_VERSION="$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "Python detecte : $PYTHON_CMD ($PYTHON_VERSION)"

case "$PYTHON_VERSION" in
    3.10*|3.11*|3.12*)
        echo "Version Python recommandee pour une compatibilite Linux elevee."
        ;;
    3.13*)
        echo "[INFO] Python 3.13 detecte. Cela peut fonctionner, mais 3.10 a 3.12 restent les versions Linux les plus previsibles."
        ;;
    *)
        echo "[INFO] Version Python non testee prioritairement. L'installation peut tout de meme reussir."
        ;;
esac

echo
echo "[2/5] Verification du support venv..."
if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
    print_venv_help
    exit 1
fi
echo "Module venv disponible."

echo
echo "[3/5] Creation de l'environnement virtuel local..."
if [[ ! -f ".venv/bin/python" ]]; then
    "$PYTHON_CMD" -m venv .venv
else
    echo "Environnement .venv deja present."
fi

echo
echo "[4/5] Mise a jour de pip et installation des dependances..."
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo
echo "[5/5] Verification rapide de l'installation..."
python -c "import flask, tensorflow, sklearn, pandas, seaborn, imblearn, PIL, joblib; print('Verification OK')"

echo
echo "==============================================================="
echo "Installation terminee avec succes."
echo
echo "Compatibilite Linux :"
echo "- Teste en priorite pour Zorin OS et distributions basees sur Ubuntu/Debian."
echo "- Generalement compatible avec Linux Mint, Pop!_OS et Debian."
echo "- Fedora / Arch / Manjaro peuvent fonctionner, mais parfois avec de legeres adaptations systeme."
echo
echo "Dataset :"
echo "- Si le dataset existe deja dans un dossier standard du projet, le notebook le trouvera automatiquement."
echo "- Sinon, le notebook tentera de le telecharger depuis Kaggle."
echo "- Si Kaggle n'est pas configure, ajoutez ~/.kaggle/kaggle.json"
echo "  ou definissez KAGGLE_USERNAME et KAGGLE_KEY."
echo
echo "Scripts disponibles :"
echo "- setup_project.sh : installation de l'environnement"
echo "- run_api.sh       : lancement de l'API Flask"
echo "- run_notebook.sh  : ouverture du notebook dans Jupyter Lab"
echo "==============================================================="
