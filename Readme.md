# Challenge Etude de Cas - Equipe 2

## Description
Projet de classification d'images utilisant Zero-Shot Learning avec CLIP et d'autres techniques avancées.

## Structure du Dossier
```
├── Readme.md                      # Documentation
├── CLIP_Class.py                   # Classification avec CLIP
├── expert_zero_shot_superclasses copy.py # Zero-Shot avancé
├── product_color_list.csv          # Liste des couleurs
├── product_list.csv                # Liste des produits
├── requirements.txt                # Dépendances
├── augmented_data/                 # Données augmentées
├── data_exploration/               # Exploration des données
├── expert_models/                  # Modèles experts
├── image_processing/               # Traitement d'images
├── inference/                      # Inférence et évaluation
├── process_new_classes/            # Gestion de nouvelles classes
├── prototypical_models/            # Modèles Zero-Shot
└── training_models/                # Entraînement
```

## Installation
```bash
git clone https://github.com/ton-repo/bodartofficiel-challenge-etude-de-cas.git
cd bodartofficiel-challenge-etude-de-cas
pip install -r requirements.txt
```

## Utilisation
- **Exploration des données** : `data_exploration/`
- **Augmentation** : `augmented_data/`
- **Classification Zero-Shot** : `CLIP_Class.py`, `expert_zero_shot_superclasses.py`
- **Inférence et évaluation** : `inference/`
- **Ajout de nouvelles classes** : `process_new_classes/`
