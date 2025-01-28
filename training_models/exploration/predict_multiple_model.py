import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image


classes_csv = pd.read_csv('product_all_classes.csv')


def predict_percentage_for_each_label(model, image_tensor, class_name, train_dataset):
    model.eval()
    all_labels = classes_csv[class_name].unique()
    with torch.no_grad():
        outputs = model(image_tensor)
        top20_values, top_20 =torch.topk(outputs, 50)
        top20_predicted_classes = [train_dataset.classes[idx][:3] for idx in top_20[0].cpu().numpy()]
        probabilities = torch.nn.functional.softmax(top20_values, dim=1)
    percentages = {top20_predicted_classes[i]: min(probabilities[0][i].item() * 100, 50) for i in range(50)}
    for label in train_dataset.classes:
        if label[:3] not in percentages:
            percentages[label[:3]] = 0.0
    return percentages

def predict_top_10_classes_for_each_label(model, image_tensor, class_name, train_dataset):
    model.eval()
    all_labels = classes_csv[class_name].unique()
    with torch.no_grad():
        outputs = model(image_tensor)
        top10_values, top_10 = torch.topk(outputs, 10)
        top10_predicted_classes = [train_dataset.classes[idx][:3] for idx in top_10[0].cpu().numpy()]
    values = [50, 30, 10, 5, 3, 2, 1, 0, 0, 0]
    percentages = {top10_predicted_classes[i]: values[i] for i in range(10)}
    for label in train_dataset.classes:
        if label[:3] not in percentages:
            percentages[label[:3]] = 0.0
    return percentages

# Example usage
model = models.resnet50(weights=True)
class_name = 'classe'
#num_classes = len(classes_csv[class_name].unique())
num_classes = 430
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_finetuned_3.pth", weights_only=True))

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data_dir = 'cropped-dataset/articles_test'
train_data_dir_color = "../organized_color_cropped_train_val/train"
train_data_dir_pepin = "reorganized_dataset/train"


test_dataset = ImageFolder(root=test_data_dir, transform=test_transform)
train_dataset_color = ImageFolder(root=train_data_dir_color, transform=test_transform)
train_dataset_pepin = ImageFolder(root=train_data_dir_pepin, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

# for images, labels in test_loader:
#     percentages = predict_percentage_for_each_label(model, images, class_name, train_dataset_pepin)
#     break


def predict_and_calculate_score(image, classes_csv, class_names, model_paths, train_datasets):
    models_dict = {}
    train_dataset_dict = {}
    for class_name, model_path, train_dataset in zip(class_names, model_paths, train_datasets):
        if class_name == 'color_class':
            num_classes = 563
        elif class_name == 'classe':
            num_classes = 430
        else:
            num_classes = len(classes_csv[class_name].unique())
        model = models.resnet50(weights=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        models_dict[class_name] = model
        train_dataset_dict[class_name] = train_dataset

    percentages_dict = {}
    for class_name in class_names:
        percentages_dict[class_name] = predict_percentage_for_each_label(models_dict[class_name], image, class_name, train_dataset_dict[class_name])

    def calculate_multiplicative_score(row):
        score = 1.0
        for class_name in class_names:
            key = str(row[class_name])  # Ensure row[class_name] is a string
            if key in percentages_dict[class_name]:  # Check if the key exists
                score += percentages_dict[class_name][key]
            # else: score = 0
            # print(class_names, row[class_name], percentages_dict[class_name].get(row[class_name], 0))
            # print(score)
        return score

    classes_csv['score'] = classes_csv.apply(calculate_multiplicative_score, axis=1)
    best_matches = classes_csv.nlargest(10, 'score')
    best_article_ids = best_matches['article_id'].tolist()
    print(best_matches[['article_id', 'score']])
    return best_article_ids

# Example usage
class_names = ['color_class', 'classe']
model_paths = ["resnet50_color_eval.pth", "resnet50_finetuned_3.pth"]
train_datasets = [train_dataset_color, train_dataset_pepin]

good_pred = [0] * 10
pred = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for class_dir in os.listdir(test_data_dir):
    class_path = os.path.join(test_data_dir, class_dir)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).unsqueeze(0)
            best_article_id = predict_and_calculate_score(image, classes_csv, class_names, model_paths, train_datasets)
            print(f"Best article_id for image {image_name[:-6]}: {best_article_id}")
            for i in range(10):
                if image_name[:-6] in best_article_id[:i+1]:
                    good_pred[i] += 1
            pred += 1

for i in range(10):
    print(f'Top {i+1} bonnes predictions par rapport au nombre total : ', good_pred[i]/pred, good_pred[i])