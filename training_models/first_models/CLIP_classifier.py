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
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# Charge le dataset
dataset_path = "augmented_data/augmented_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset(str(dataset_path), trust_remote_code=True, num_augments = 0)

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
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        
            
        
        return image, label

# Créer le dataset
test_dataset = CustomDataset(dataset['test'], transform=transform)


# Créer les dataloaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

label_mapping = {
            0: "Glasses",
            1: "Woman Bag",
            2: "Shoes",
            3: "Watch",
            
        }


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def evaluate_clip(model, processor, val_dataset, label_mapping, device):
    """
    Évalue le modèle CLIP sur un jeu de données de validation.
    
    Arguments :
        model : CLIPModel pré-entraîné.
        processor : CLIPProcessor associé.
        val_dataset : Dataset PyTorch contenant les images et labels.
        label_mapping : Dictionnaire mapant les labels textuels vers des descriptions.
        device : Périphérique PyTorch ('cuda' ou 'cpu').
    
    Retourne :
        accuracy : Précision du modèle sur le jeu de validation.
    """
    model.eval()  # Mode évaluation
    all_preds = []
    all_labels = []

    print(list(set(label_mapping.values())))
    with torch.no_grad():
        for images, labels in tqdm(val_dataset):
            
            # Préparation des entrées
            inputs = processor(
                text=list(label_mapping.values()), 
                images=images, 
                return_tensors="pt", 
                padding=True,
                do_rescale = False
            ).to(device)
            
            # Calcul des similarités image-texte
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Similarité image-texte
            print(logits_per_image)
            probs = logits_per_image.softmax(dim=1)
            # Prédictions (classe avec la probabilité maximale)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            print(preds)
            # Convertir les labels en index pour comparaison
            
            all_labels.extend(labels)

    # Calcul de la précision
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

print(evaluate_clip(model, processor, test_loader, label_mapping, device))