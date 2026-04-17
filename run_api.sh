#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Lance l'API Flask du projet sous Linux.
# Si l'environnement virtuel n'existe pas encore, on installe d'abord.
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

echo "Lancement de l'API Flask..."
echo "Ouvrez ensuite http://127.0.0.1:5000 dans votre navigateur."
python api_flask_pneumonia.py
