#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Ouvre le notebook principal dans Jupyter Lab sous Linux.
# Le notebook gere automatiquement la recherche du dataset et tente
# un telechargement Kaggle si necessaire.
# ================================================================

cd "$(dirname "$0")"

detect_support_level() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        local distro_id="${ID:-}"
        local distro_like=" ${ID_LIKE:-} "

        case "$distro_id" in
            zorin|ubuntu|linuxmint|pop|debian) echo "supportee"; return ;;
            fedora|arch|manjaro) echo "probable"; return ;;
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

SUPPORT_LEVEL="$(detect_support_level)"
case "$SUPPORT_LEVEL" in
    supportee) echo "[Compatibilite Linux] Distribution supportee." ;;
    probable) echo "[Compatibilite Linux] Support probable : quelques ajustements systeme peuvent etre necessaires." ;;
    *) echo "[Compatibilite Linux] Distribution non testee : execution possible mais non garantie." ;;
esac

if [[ ! -f ".venv/bin/python" ]]; then
    echo "Environnement Python non detecte. Installation en cours..."
    bash ./setup_project.sh
fi

source .venv/bin/activate

echo "Ouverture du notebook principal..."
python -m jupyter lab "Projet Science des donnees/Classification_CNN_Pneumonia_OPTIMIZED.ipynb"
