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

# Charge le dataset
dataset_path = "augmented_dataset_expert"
num_augment = 2
num_epochs = 5
margin = 1
lr = 1e-4
for class_names in ["W Shoes", "W Accessories", "W SLG", "Watches", "W Bags"]:

    dataset = load_dataset(str(dataset_path), trust_remote_code=True, num_augments = num_augment , class_name = class_names)

    from sklearn.preprocessing import LabelEncoder
    import json


    # Créer un encodeur
    label_encoder = LabelEncoder()

    # Encoder les labels de l'ensemble de données "train"
    train_labels = dataset['train']['label']  # Assurez-vous que les labels sont sous forme de liste ou tableau
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    # Save label encoder
    label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
    dataset['train'] = dataset['train'].add_column('encoded_label', encoded_train_labels)
    with open(f'label_encoder_{dataset_path.split("_")[-1]}.txt', 'w') as f:
        json.dump(label_mapping, f)


    test_labels = dataset['test']['label']
    encoded_test_labels = label_encoder.transform(test_labels)

    dataset['test'] = dataset['test'].add_column('encoded_label', encoded_test_labels)


    # Vérifier la disponibilité du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Définir les transformations pour les images
    transform = transforms.Compose([
       # ViT prend généralement des images de taille (224, 224)
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

    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Charger le modèle Vision Transformer (ViT) pré-entraîné
    
    class TripletLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(TripletLoss, self).__init__()
            self.margin = margin

        def forward(self, anchor, positive, negative):
            # Calculer les distances
            pos_distance = F.pairwise_distance(anchor, positive)
            neg_distance = F.pairwise_distance(anchor, negative)
            
            # Calculer la triplet loss
            losses = F.relu(pos_distance - neg_distance + self.margin)
            return losses.mean()
    
    def compute_prototypes(features, labels):
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            # Calculer la moyenne des embeddings pour chaque classe
            class_features = features[labels == label]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes), unique_labels

    def prototypical_loss(features, labels, prototypes, prototype_labels):
        # Calculer les distances entre les features et les prototypes
        distances = torch.cdist(features, prototypes)
        
        # Convertir les labels en indices pour les prototypes
        label_to_index = {label.item(): idx for idx, label in enumerate(prototype_labels)}
        target_indices = torch.tensor([label_to_index[label.item()] for label in labels]).to(device)
        
        # Calculer la loss (negative log-likelihood)
        loss = F.cross_entropy(-distances, target_indices)
        return loss
    
    
    
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
                correct = preds.eq(labels.view(-1, 1).expand_as(preds))  # Compare with labels
                top_k_correct += correct.sum().item()  # Sum the number of correct top-k predictions
                total_samples += labels.size(0)  # Count the total number of samples

        # Calculate the top-k accuracy
        top_k_accuracy = top_k_correct / total_samples
        return top_k_accuracy

    print(device)
    # Définir la fonction de perte et l'optimiseur
    def generate_triplets(features, labels):
        anchors = []
        positives = []
        negatives = []
        
        for i in range(len(features)):
            anchor = features[i]
            label = labels[i]
            
            # Trouver un positif (même classe)
            positive_indices = (labels == label).nonzero(as_tuple=True)[0]
            positive_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))]
            positive = features[positive_idx]
            
            # Trouver un négatif (classe différente)
            negative_indices = (labels != label).nonzero(as_tuple=True)[0]
            negative_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
            negative = features[negative_idx]
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    
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

        # Définir le modèle et la loss
    model = models.mobilenet_v2(pretrained = True)
    model.classifier = nn.Identity()  # Supprimer la dernière couche de classification
    model = model.to(device)

    triplet_loss_fn = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Boucle d'entraînement
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Extraire les features
            features = model(images)
            
            # Générer des triplets (anchor, positive, negative)
            # (Vous pouvez utiliser une fonction pour générer des triplets)
            anchors, positives, negatives = generate_triplets(features, labels)
            
            # Calculer la triplet loss
            triplet_loss = triplet_loss_fn(anchors, positives, negatives)
            
            # Calculer les prototypes et la prototypical loss
            prototypes, prototype_labels = compute_prototypes(features, labels)
            proto_loss = prototypical_loss(features, labels, prototypes, prototype_labels)
            
            # Combiner les losses
            loss = triplet_loss + proto_loss
            
            # Mise à jour du modèle
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        
    # Create test dataset and dataloader
    test_dataset = CustomDataset(dataset['test'], transform=transform)
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
                features.append(outputs.cpu())  # Déplacer les features sur le CPU
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
    def predict_labels(similarity_matrix, train_labels, top_k=1):
        _, indices = torch.topk(similarity_matrix, top_k, dim=1)
        predicted_labels = train_labels[indices]
        if top_k > 1:
            predicted_labels = torch.mode(predicted_labels, dim=1).values
        return predicted_labels

    # Prédire les labels pour l'ensemble de test
    predicted_labels = predict_labels(similarity_matrix, train_labels, top_k=1)

    # Calculer l'accuracy
    accuracy = accuracy_score(test_labels.numpy(), predicted_labels.numpy())
    # Predict top-3 labels for test set
    # Suppose `similarity_matrix` is your matrix of predicted similarities (test x train)
# and `train_labels` contains the corresponding labels for the training embeddings.

    def predict_labels(similarity_matrix, train_labels, top_k=3):
        # Trouver les indices des top_k similarités (par ordre décroissant)
        top_k_indices = similarity_matrix.topk(top_k, dim=1, largest=True).indices
        # Obtenir les labels prédits correspondants
        top_k_predicted_labels = train_labels[top_k_indices]
        return top_k_predicted_labels

    # Exemple : prédictions pour les 3 premiers
    top_3_predicted = predict_labels(similarity_matrix, train_labels, top_k=3)

    # Vérifier si le vrai label est à la position 1, 2 ou 3
    correct_top_3 = (test_labels == top_3_predicted[:, 0]) | \
                    (test_labels == top_3_predicted[:, 1]) | \
                    (test_labels == top_3_predicted[:, 2])

    # Calculer la précision
    accuracy_top_3 = correct_top_3.float().mean().item()

    print(f"Zero-shot accuracy: {accuracy:.4f}", f"Top-3 accuracy: {accuracy_top_3:.4f}")