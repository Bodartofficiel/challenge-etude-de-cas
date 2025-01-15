#import modules and datasets
import pandas as pd
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pathlib import Path

# Charge le dataset
dataset_path = "augmented_dataset_expert"
dataset = load_dataset(str(dataset_path), trust_remote_code=True, num_augments=7, class_name="W Shoes")

from sklearn.preprocessing import LabelEncoder

# Créer un encodeur
label_encoder = LabelEncoder()

# Encoder les labels de l'ensemble de données "train"
train_labels = dataset['train']['label']  # Assurez-vous que les labels sont sous forme de liste ou tableau
encoded_train_labels = label_encoder.fit_transform(train_labels)

dataset['train'] = dataset['train'].add_column('encoded_label', encoded_train_labels)
test_labels = dataset['test']['label']
encoded_test_labels = label_encoder.transform(test_labels)

dataset['test'] = dataset['test'].add_column('encoded_label', encoded_test_labels)

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définir les transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Créer une classe Dataset personnalisée
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['encoded_label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Créer le dataset
train_dataset = CustomDataset(dataset['train'], transform=transform)

# Diviser le dataset en train et validation (10% pour validation)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])

# Créer les dataloaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Charger le modèle EfficientNet pré-entraîné (par exemple EfficientNet-B0)
model = EfficientNet.from_pretrained('efficientnet-b0')

# Gel des poids des couches initiales (optionnel si vous voulez fine-tuner toutes les couches)
for param in model.parameters():
    param.requires_grad = False

# Modifier la dernière couche pour correspondre au nombre de classes de votre dataset
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(set(dataset['train']['encoded_label'])))  # Utiliser le nombre de classes dans votre dataset

# Déplacer le modèle sur le GPU
model = model.to(device)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model._fc.parameters(), lr=0.001)  # Optimiser uniquement les poids de la dernière couche

# Fonction pour évaluer le modèle
def evaluate(model, val_loader):
    model.eval()  # Mettre le modèle en mode évaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)  # Déplacer les images vers le GPU
            labels = labels.to(device)  # Déplacer les labels vers le GPU
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Entraîner le modèle
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    model.train()  # Mettre le modèle en mode entraînement
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)  # Déplacer les images vers le GPU
        labels = labels.to(device)  # Déplacer les labels vers le GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Calculer l'accuracy sur l'ensemble de validation
    val_accuracy = evaluate(model, val_loader)
    
    # Afficher la perte et l'accuracy
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Accuracy: {val_accuracy}')

print('Finished Training')

# Create test dataset and dataloader
test_dataset = CustomDataset(dataset['test'], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate on test set
model.eval()
test_accuracy = evaluate(model, test_loader)
print(f'Test Accuracy: {test_accuracy}')
