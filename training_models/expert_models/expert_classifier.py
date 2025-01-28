import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json
# Charge le dataset
dataset_path = Path(__file__).parent.parent.parent / "augmented_data/augmented_dataset_expert"

for class_names in ["W Accessories", "W SLG", "Watches"]:

    dataset = load_dataset(str(dataset_path), trust_remote_code=True, num_augments = 0, class_name = class_names)

    


    # Créer un encodeur
    label_encoder = LabelEncoder()

    # Encoder les labels de l'ensemble de données "train"
    train_labels = dataset['train']['label']  # Assurez-vous que les labels sont sous forme de liste ou tableau
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    # Save label encoder
    label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
    dataset['train'] = dataset['train'].add_column('encoded_label', encoded_train_labels)
    with open(f'label_encoder_{class_names}.txt', 'w') as f:
        json.dump(label_mapping, f)
