import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os
from evaluate import load

f1_score = metric = load("f1")
# Paths to the new dataset
train_data_dir = "../organized_color_cropped_train_val/train"
val_data_dir = "../organized_color_cropped_train_val/val"

# Define transforms for augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=train_data_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_data_dir, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


# Load ResNet-50
model = models.resnet50(pretrained=True)

# Replace the fully connected layer to match the new number of classes
num_classes = len(train_dataset.classes)  # Automatically gets the number of 
print("number of classes:", num_classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_and_save(model=model, num_epochs=45, output_path="resnet50_finetuned.pth"):
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer (use AdamW for better performance with fine-tuning)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Scheduler to reduce learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final layers
    for param in model.fc.parameters():
        param.requires_grad = True


    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler
        scheduler.step()

    torch.save(model.state_dict(), output_path)


def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_predictions_top3 = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_top3 = torch.topk(outputs, 3)
            
            predicted = predicted_top3[:,0]
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            for i in range(len(labels)):
                labeli = labels[i].cpu().numpy().tolist()
                if any(predicted_top3[i,:] == labeli):
                    all_predictions_top3.append(labeli)
                else:
                    all_predictions_top3.append(predicted_top3[i,0].cpu().numpy().tolist())

    val_f1_score = f1_score.compute(predictions=all_predictions, references=all_labels, average='weighted')['f1']
    print(f"Validation F1 Score: {val_f1_score:.4f}")
    print(f"Validation F1 Score top 3: {f1_score.compute(predictions=all_predictions_top3, references=all_labels, average='weighted')['f1']:.4f}")
    return val_f1_score



def predict_and_save_to_csv(model_path="resnet50_finetuned.pth", test_images_dir="cropped-dataset/articles_test", output_csv="predictions.csv", train_dataset=train_dataset):
    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # List to store predictions
    predictions = []

    # Iterate over test images
    for class_dir in os.listdir(test_images_dir):
        class_path = os.path.join(test_images_dir, class_dir)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path).convert('RGB')
                image = test_transform(image).unsqueeze(0).to(device)

                # Predict the top 3 classes
                with torch.no_grad():
                    outputs = model(image)
                    _, top3_predicted = torch.topk(outputs, 3)
                    top3_predicted_classes = [train_dataset.classes[idx] for idx in top3_predicted[0].cpu().numpy()]
                    predicted_class = top3_predicted_classes[0]  # Use the top-1 prediction for the main prediction
                    predicted_second_class = top3_predicted_classes[1]
                    predicted_third_class = top3_predicted_classes[2]

                # Remove the .jpg extension from the image name
                image_name_no_ext = os.path.splitext(image_name)[0][:-2]

                # Append to predictions list
            predictions.append({
                    "image_name": image_name_no_ext, 
                    "predicted_class": predicted_class[:-6], 
                    "predicted_second_class": predicted_second_class[:-6], 
                    "predicted_third_class": predicted_third_class[:-6]
                })
                # Load the product list with new classes CSV
                product_list_df = pd.read_csv("product_color_list.csv")

                # Merge predictions with labeled test articles to get the real article_id
                predictions_df = pd.DataFrame(predictions)
                
                # Merge with product list to get real_class and group_class
                predictions_df = predictions_df.merge(product_list_df, left_on="image_name", right_on="MMC", how="left")

                # Save the final predictions to CSV
                predictions_df.to_csv(output_csv, index=False)
    # Calculate the percentage of correct predictions
    correct_predictions = 0
    correct_predictions_second = 0
    correct_predictions_third = 0
    # Print the first five values of predicted_class, classe and the assertion that they are equal
    for _, row in predictions_df.iterrows():
        print(f"Predicted: {row['predicted_class']}, Actual: {row['color_class']}, Equal: {row['predicted_class'] == str(row['color_class'])}")
        if row['predicted_class'] == str(row['color_class']):
            correct_predictions += 1
        if row['predicted_second_class'] == str(row['color_class']):
            correct_predictions_second += 1
        if row['predicted_third_class'] == str(row['color_class']):
            correct_predictions_third += 1
    print('correct_predictions:', correct_predictions)
    print('correct_predictions_second:', correct_predictions_second)
    print('correct_predictions_third:', correct_predictions_third)
    total_correct_predictions = correct_predictions + correct_predictions_second + correct_predictions_third
    print('total correct predictions:', total_correct_predictions)
        
    total_predictions = len(predictions_df)
    accuracy_percentage = 100 * total_correct_predictions / total_predictions
    print(f"Percentage of exact predictions: {100 * correct_predictions / total_predictions:.2f}%")
    print(f"Percentage of top 3 predictions: {accuracy_percentage:.2f}%")
                

model.load_state_dict(torch.load("resnet50_color_eval.pth"))
model.to(device)

val_f1_score = evaluate_model(model, val_loader, device)

predict_and_save_to_csv("resnet50_color_eval.pth", "cropped-dataset/articles_test", "predictions_color.csv", train_dataset)
