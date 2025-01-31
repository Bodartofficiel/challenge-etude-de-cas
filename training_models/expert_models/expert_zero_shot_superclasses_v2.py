import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pathlib import Path
import torch.nn.functional as F

results = pd.DataFrame(
    columns=[
        "superclass",
        "class",
        "zero-shot accuracy",
        "top-3 accuracy",
        "number of test images",
        "number of possible articles",
    ]
)
# Charge le dataset
dataset_path = "augmented_data/augmented_dataset_expert"
num_augment = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for class_names in ["W Shoes", "W Accessories", "W SLG", "Watches", "W Bags"]:
    dataset = load_dataset(
        str(dataset_path),
        trust_remote_code=True,
        num_augments=num_augment,
        class_name=class_names,
    )

    from sklearn.preprocessing import LabelEncoder
    import json

    # Créer un encodeur
    label_encoder = LabelEncoder()

    # Encoder les labels de l'ensemble de données "train"
    train_labels = dataset["train"]["label"]  # Assurez-vous que les labels sont sous forme de liste ou tableau
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    # Save label encoder
    label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
    dataset["train"] = dataset["train"].add_column(
        "encoded_label", encoded_train_labels
    )
    with open(f"label_encoder_{dataset_path.split('_')[-1]}.txt", "w") as f:
        json.dump(label_mapping, f)

    test_labels = dataset["test"]["label"]
    encoded_test_labels = label_encoder.transform(test_labels)

    dataset["test"] = dataset["test"].add_column("encoded_label", encoded_test_labels)

    # Charger le modèle Vision Transformer (ViT) pré-entraîné
    

    import timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # Supprimer la dernière couche de classification
    model.classifier = nn.Identity()

    # Vérifier la disponibilité du GPU

    # Définir les transformations pour les images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Créer une classe Dataset personnalisée
    class CustomDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]["image"]
            label = self.dataset[idx]["encoded_label"]
            if self.transform:
                image = self.transform(image)
            return image, label

    # Créer le dataset
    train_dataset = CustomDataset(dataset["train"], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    def evaluatetop3(model, val_loader, top_k=3):
        model.eval()  # Set the model to evaluation mode
        top_k_correct = 0  # Initialize counter for correct top-k predictions
        total_samples = 0  # Initialize total samples counter

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).logits

                # Get the top-k predictions
                _, preds = torch.topk(outputs, top_k, dim=1, largest=True, sorted=True)

                # Check if the true label is among the top-k predictions
                correct = preds.eq(
                    labels.view(-1, 1).expand_as(preds)
                )  # Compare with labels
                top_k_correct += (
                    correct.sum().item()
                )  # Sum the number of correct top-k predictions
                total_samples += labels.size(0)  # Count the total number of samples

        # Calculate the top-k accuracy
        top_k_accuracy = top_k_correct / total_samples
        return top_k_accuracy

    model = model.to(device)
    print(device)
    # Définir la fonction de perte et l'optimiseur

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

    # Create test dataset and dataloader
    test_dataset = CustomDataset(dataset["test"], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Évaluer sur l'ensemble de test
    model.eval()

    # Supprimer la dernière couche de classification

    # Fonction pour extraire les features
    def extract_features(model, dataloader):
        features = []
        labels = []
        with torch.no_grad():  # Désactiver le calcul du gradient
            for images, lbls in dataloader:
                images = images.to(device)  # Déplacer les images sur le GPU
                outputs = model(images)  # Extraire les features
                try:
                    features.append(
                        outputs.logits.cpu()
                    )  # Déplacer les features sur le CPU
                except:
                    features.append(outputs.cpu())
                labels.append(lbls.cpu())  # Déplacer les labels sur le CPU
        return torch.cat(features), torch.cat(labels)

    # Extraire les features pour l'ensemble d'entraînement et de test
    train_features, train_labels = extract_features(model, train_loader)
    test_features, test_labels = extract_features(model, test_loader)

    # Calculer la similarité cosinus entre les features de test et d'entraînement
    def cosine_similarity_matrix(a, b):
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.t())

    similarity_matrix = cosine_similarity_matrix(test_features, train_features)

    # Prédire les labels en fonction de la similarité cosinus

    # Prédire les labels en fonction de la similarité cosinus
    def predict_labels(similarity_matrix, train_labels, top_k=1):
        _, indices = torch.topk(similarity_matrix, top_k, dim=1)
        predicted_labels = train_labels[indices]

        return predicted_labels

    # Prédire les labels pour l'ensemble de test
    predicted_labels = predict_labels(similarity_matrix, train_labels, top_k=1)

    # Calculer l'accuracy
    accuracy = accuracy_score(test_labels.numpy(), predicted_labels.numpy())

    # Predict top-3 labels for test set
    accuracy_top = []
    for k in range(2, 11):
        try:
            top_3_predicted = predict_labels(similarity_matrix, train_labels, top_k= k )
            correct_top_3 = (test_labels.unsqueeze(1) == top_3_predicted).any(dim=1)
            accuracy_top_3 = correct_top_3.float().mean().item()
        except Exception as e:
            accuracy_top_3 = "No data"
            print(f"Error calculating top-3 accuracy: {e}")
        accuracy_top.append(accuracy_top_3)
    # Calculate if true label is among top 3 predictions
    print(f"Zero-shot accuracy: {accuracy:.4f}", f"Top-3 accuracy: {accuracy_top_3}")
    new_row = {
        "superclass": class_names,
        "zero-shot accuracy": accuracy,
        "top-2 accuracy": accuracy_top[0],
        "top-3 accuracy": accuracy_top[1],
        "top-4 accuracy": accuracy_top[2],
        "top-5 accuracy": accuracy_top[3],
        "top-6 accuracy": accuracy_top[4],
        "top-7 accuracy": accuracy_top[5],
        "top-8 accuracy": accuracy_top[6],
        "top-9 accuracy": accuracy_top[7],
        "top-10 accuracy": accuracy_top[8],
        "number of test images": len(dataset["test"]),
        "number of possible articles": len(dataset["train"]),
    }
    results = pd.concat([results, pd.DataFrame([new_row])])
    results.to_csv("results_ViT_top_k_superclasses.csv")
