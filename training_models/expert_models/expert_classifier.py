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
# Charge le dataset
dataset_path = Path(__file__).parent.parent.parent / "augmented_data/augmented_dataset_expert"

dataset = load_dataset(str(dataset_path), trust_remote_code=True, num_augments = 4, class_name = "Watches" )

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


# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définir les transformations pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT prend généralement des images de taille (224, 224)
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

# Charger le modèle Vision Transformer (ViT) pré-entraîné
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(set(dataset['train']['encoded_label'])))

# Geler tous les paramètres du modèle sauf la dernière couche
for param in model.parameters():
    param.requires_grad = False

# Débloquer les paramètres de la dernière couche (classifier)
for param in model.classifier.parameters():
    param.requires_grad = True

def evaluatetop3(model, val_loader, top_k=3):
    model.eval()  # Set the model to evaluation mode
    top_k_correct = 0  # Initialize counter for correct top-k predictions
    total_samples = 0  # Initialize total samples counter

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits

            # Get the top-k predictions
            _, preds = torch.topk(outputs, top_k, dim=1, largest=True, sorted=True)
            
            # Check if the true label is among the top-k predictions
            correct = preds.eq(labels.view(-1, 1).expand_as(preds))  # Compare with labels
            top_k_correct += correct.sum().item()  # Sum the number of correct top-k predictions
            total_samples += labels.size(0)  # Count the total number of samples

    # Calculate the top-k accuracy
    top_k_accuracy = top_k_correct / total_samples
    return top_k_accuracy

model = model.to(device)
print(device)
# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimiser tous les paramètres du modèle

# Fonction pour évaluer le modèle
def evaluate(model, val_loader):
    model.eval()  # Mettre le modèle en mode évaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)  # Déplacer les images vers le GPU
            labels = labels.to(device)  # Déplacer les labels vers le GPU
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Entraîner le modèle
num_epochs = 5
for epoch in tqdm(range(num_epochs)):
    model.train()  # Mettre le modèle en mode entraînement
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)  # Déplacer les images vers le GPU
        labels = labels.to(device)  # Déplacer les labels vers le GPU
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Calculer l'accuracy sur l'ensemble de validation
    val_accuracy = evaluate(model, val_loader)
    val_accuracytop3 = evaluatetop3(model, val_loader, 3)
    # Afficher la perte et l'accuracy
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Accuracy: {val_accuracy}, Val top3 Accuracy {val_accuracytop3}')

print('Finished Training')

# Create test dataset and dataloader
test_dataset = CustomDataset(dataset['test'], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Évaluer sur l'ensemble de test
model.eval()
test_accuracy = evaluate(model, test_loader)
testtop3 = evaluatetop3(model, test_loader, 3)
print(f'Test Accuracy: {test_accuracy}, top3: {testtop3}')
