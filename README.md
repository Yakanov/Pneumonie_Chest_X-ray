# Pneumonie_Chest_X-ray
Développer un système d’intelligence artificielle capable d’analyser des radiographies thoraciques (chest X-ray) et de prédire automatiquement si une personne est atteinte de pneumonie ou si ses poumons sont normaux.

## Detection de Pneumonie sur Radiographies Thoraciques (CNN + Ensemble + XAI)

Projet de Science des Donnees pour la detection binaire NORMAL vs PNEUMONIA a partir de radiographies thoraciques.

Ce projet combine:
- Deep Learning (EfficientNet B0 + ResNet50)
- Machine Learning classique (Gradient Boosting)
- Ensemble Voting (majorite)
- XAI avec GRAD-CAM
- API Flask avec interface web

## Objectifs
- Detecter automatiquement la pneumonie a partir d'une image thoracique.
- Expliquer visuellement les zones d'attention du modele (GRAD-CAM).
- Fournir une solution reproductible et portable (Windows + Linux).

## Fonctionnalites principales
- Prediction via API Flask (upload image)
- Mode lot (plusieurs images d'un coup)
- Affichage GRAD-CAM
- Filtrage d'images non thoraciques (rejet des images hors contexte radiologique)
- Focalisation GRAD-CAM sur les zones pulmonaires (masque strict)
- Scripts de lancement automatiques

## Structure utile
- api_flask_pneumonia.py: API principale
- Projet Science des donnees/Classification_CNN_Pneumonia_OPTIMIZED.ipynb: notebook principal
- Projet Science des donnees/saved_models/: artefacts modeles
- setup_project.bat / run_api.bat / run_notebook.bat: scripts Windows
- setup_project.sh / run_api.sh / run_notebook.sh: scripts Linux
- GUIDE_INSTALLATION_WINDOWS_LINUX.txt: guide rapide
- portable_share.zip: package partage pret a envoyer

## Dataset
Dataset utilise:
- Chest X-Ray Images (Pneumonia)
- Source Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Le notebook et l'API tentent de localiser automatiquement le dataset localement.
En cas d'absence, le telechargement Kaggle peut etre tente si la configuration Kaggle est disponible.

## Prerequis
- Python 3.10 a 3.12 recommande (3.13 possible selon environnement)
- pip
- Connexion internet (si telechargement automatique de dependances ou dataset)

## Installation et execution (Windows)
1. Lancer setup_project.bat
2. Lancer run_api.bat pour demarrer l'API
3. Ouvrir http://127.0.0.1:5000
4. Ou lancer run_notebook.bat pour travailler depuis le notebook

## Installation et execution (Linux)
1. Ouvrir un terminal a la racine du projet
2. Executer: bash setup_project.sh
3. Executer: bash run_api.sh
4. Ouvrir http://127.0.0.1:5000
5. Ou executer: bash run_notebook.sh

Compatibilite Linux:
- Supporte en priorite: Zorin OS, Ubuntu, Linux Mint, Pop!_OS, Debian
- Support probable: Fedora, Arch, Manjaro
- Les scripts affichent un diagnostic automatique au lancement

## API - endpoint principal
- POST /predict-upload-xai
  - Entree: fichier image
  - Sortie: prediction, probabilites, votes, image GRAD-CAM (base64), statut XAI

## Methodologie modele
1. Pretraitement image
2. Predictions independantes:
   - EfficientNet
   - ResNet50
   - Gradient Boosting
3. Vote majoritaire
4. Explicable via GRAD-CAM
5. Filtrage et focalisation pulmonaire stricte pour limiter les zones non pertinentes

## Bonnes pratiques de partage
Pour partager facilement le projet avec un camarade:
1. Envoyer portable_share.zip
2. Le camarade dezippe
3. Il lance setup_project.bat (Windows) ou setup_project.sh (Linux)
4. Il lance run_api.bat / run_api.sh

## Auteurs
Projet de groupe realise dans le cadre du module de Science des Donnees.

# Avertissement
Ce projet est a but academique et pedagogique.
Il ne remplace pas un diagnostic medical professionnel.

## Sauvegarde du Modele
- le dossier de sauvegarde du modele depassant les 100Mo, nous ne pouvions pas le pusher sur git. En dessous vous avez le lien Google drive pour le recuperer.
- https://drive.google.com/drive/folders/1fhYd9M7x5iGOnW1axrJ6rtJDXxtCTZuT?usp=drive_link
- NB: ce dossier est necessaire pour faire fonctionner l'API
