# imports 
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing


def prepare_data():
    image_train_directory = './data/articles_train'
    image_test_directory = './data/articles_test'

    print(os.listdir(image_train_directory))
    print(os.listdir(image_test_directory))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train = datasets.ImageFolder(image_train_directory, data_transforms)
    dataset_test = datasets.ImageFolder(image_test_directory, data_transforms)

    np.random.seed(42)
    labels = [label for _, label in dataset_train.samples]
    samples_train, samples_val = train_test_split(
        dataset_train.samples, 
        test_size=0.20, 
        stratify=labels
    )

    # Datasets
    dataset_train_split = datasets.ImageFolder(image_train_directory, data_transforms)
    dataset_train_split.samples = samples_train
    dataset_train_split.imgs = samples_train
    
    dataset_val = datasets.ImageFolder(image_train_directory, data_transforms)
    dataset_val.samples = samples_val
    dataset_val.imgs = samples_val

    dataset_test = datasets.ImageFolder(image_test_directory, data_transforms)

    # DataLoader avec num_workers ajusté
    loader_train = torch.utils.data.DataLoader(
        dataset_train_split, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0  # Réduit à 0 pour éviter les problèmes de multiprocessing
    )

    return loader_train, dataset_val, dataset_test

def evaluate(model, dataset, criterion, device):
    avg_loss = 0.
    avg_accuracy = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()  # Met le modèle en mode évaluation
    with torch.no_grad():  # Désactive le calcul des gradients pendant l'évaluation
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            n_correct = torch.sum(preds == labels)
            
            avg_loss += loss.item()
            avg_accuracy += n_correct
            
    return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)

PRINT_LOSS = True
def train_model(model, loader_train, data_val, optimizer, criterion, device, n_epochs=10):
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")
        model.train()  # Met le modèle en mode entraînement
        for i, data in enumerate(loader_train):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            if PRINT_LOSS:
                loss_val, accuracy = evaluate(model, data_val, criterion, device)
                print(f"{i} loss train: {loss.item():1.4f}\t val {loss_val:1.4f}\tAcc (val): {accuracy:.1%}")
            
            loss.backward()
            optimizer.step()

def save_model(model, path):
    torch.save(model.state_dict(), path)


def main():
    # Préparation des données
    loader_train, dataset_val, dataset_test = prepare_data()

    # Configuration du device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialisation du modèle
    print("Récupération du ResNet-18 pré-entraîné...")
    my_net = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    
    # Gel des paramètres
    for param in my_net.parameters():
        param.requires_grad = False
    
    my_net.fc = nn.Linear(in_features=512, out_features=6, bias=True)
    my_net.to(device)
    
    # Critère et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_net.fc.parameters(), lr=0.001, momentum=0.9)
    
    print("\nApprentissage en transfer learning")
    
    # Entraînement
    train_model(my_net, loader_train, dataset_val, optimizer, criterion, device, n_epochs=1)
    
    # Save the model
    save_model(my_net, 'resnet18_finetuned.pth')
    
    # Évaluation finale
    my_net.eval()
    loss, accuracy = evaluate(my_net, dataset_test, criterion, device)
    print(f"Accuracy (test): {100 * accuracy:.1f}%")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()