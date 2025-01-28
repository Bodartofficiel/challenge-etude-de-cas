import torch
from torchvision import transforms
from PIL import Image
import os
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import sys
import os
# Ajouter le chemin du répertoire parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expert_models.mapping import MAPPING_REF

from collections import Counter

'''
TODO to make this script runnable

- Download and put the weights of the expert models in the expert_models/weights folder with
this nomenclature : model_weight_W_Accessories.pth

- Download and put the checkpoint of the general model in the repository and adapt the path

- Change the mapping if necessary
'''

from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)
import numpy as np

# Define the device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")

# Set up the seed 
torch.manual_seed(0)

# Load the test dataset
def load_dataset_train_test(dataset_path):
    dataset = load_dataset(
        str(dataset_path),
        name="full_augmented_dataset_with_0_augment",
        num_augments=0,
        trust_remote_code=True,
    )
    return dataset

# Load the general model
def load_general_model(model_path):
    model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()
    processor = ViTImageProcessor.from_pretrained(model_path)
    # model.eval()
    return model, processor

# Functions to get all the paths for 
def get_expert_model_paths(expert_model_dir, labels):
    expert_model_paths = {}
    weights_path = os.path.join(expert_model_dir, 'weights')
    for chkpt in os.listdir(weights_path):
        if chkpt.endswith(".pth"):
            
            chkpt_path = os.path.join(weights_path, chkpt)
            class_name = chkpt.split('_')[-1].split('.')[0]
            if class_name != 'Watches':
                class_name= "W " + class_name
            class_idx = labels[class_name]
            expert_model_paths[class_idx] = chkpt_path
            
    return expert_model_paths

# TODO check the weights of the last layers, finetuning ?
def load_expert_models(expert_model_paths, nb_ref_per_class):
    expert_models = {}
    
    for key in expert_model_paths.keys():
    
        path = expert_model_paths[key]
        
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=nb_ref_per_class[key]
        ).to(device).eval()

        # Weight paths
        pth_weights_path = path

        state_dict = torch.load(pth_weights_path, map_location = device)

        model.load_state_dict(state_dict, strict=False)

        expert_models[key] = model
        
    return expert_models

# Predict the classes
def predict_general_model(model: ViTForImageClassification, processor, test_dataset, top_k):
    predictions = []
    with torch.no_grad():
        for image in test_dataset["image"]:

            inputs = torch.tensor(
                np.array(processor(image)["pixel_values"]), device=device
            )
            outputs = model.forward(pixel_values=inputs)
            logits = outputs.logits
            topk_indices = torch.topk(logits, top_k).indices.cpu().numpy().tolist()[0]
            predictions.append(topk_indices)
    return predictions


# Predict the articles
def get_final_predictions(expert_models, class_predictions, test_dataset, mapping, top_k):
    
    final_predictions = []
    
    for idx, class_pred in enumerate(class_predictions):
        # Get the predicted class (first element of the prediction)
        #TODO add the possibility to have multiples classes, and then select the top3 ref for each class ?
        predicted_class = class_pred[0]
        
        # Get the corresponding expert model
        expert_model = expert_models[predicted_class]
        
        # Get the name of the class
        predicted_class_name = mapping[predicted_class]
        
        # Get the image from the dataset
        image = test_dataset[idx]
        
        # Predict the reference using the expert model
        reference_predictions = predict_expert_model(
            model=expert_model, image=image, top_k=top_k
        )
        
        # Get the reference code from the mapping
        ref_pred_codes = []
        for ref_pred in reference_predictions:
            ref_pred_codes.append(MAPPING_REF[predicted_class][ref_pred])
    
        # For the basic classes
        true_class = test_dataset[idx]['path'].split('/')[2]
        
        true_reference = test_dataset[idx]['path'].split('/')[-1].split('.')[0].split('_')[0]
        
        final_predictions.append({
            'image_idx': idx,
            'true_class': true_class,
            'predicted_class': predicted_class_name,
            'true_reference' : true_reference,
            'predicted_reference': ref_pred_codes
        })
    
    return final_predictions


# Predict the article's reference
def predict_expert_model(model, image, top_k=2):
    with torch.no_grad():
        
        # TODO is it the right transformation ?
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor()
        ])
        
        image = transform(image['image'])
        image = image.unsqueeze(0).to(device)
        output = model(image)
        logits = output.logits
        values, pred = torch.max(logits, 1)
        topk_indices = torch.topk(logits, top_k).indices.cpu().numpy().tolist()[0]

    return topk_indices


def get_class_scores(predictions):
    
    # Prediction results
    y_true = [item['true_class'] for item in predictions]
    y_pred = [item['predicted_class'] for item in predictions]

    # Classification report
    print("\nClassification report (super class):")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    
def get_ref_scores(predictions):
    # Prediction results
    y_true = [item['true_reference'] for item in predictions]
    y_pred = []

    for item in predictions:
        true_ref = item['true_reference']
        predicted_refs = item['predicted_reference']

        # Compute accuracy
        # If the reference is in the predicted references, count it as a correct prediction
        if isinstance(predicted_refs, list) and true_ref in predicted_refs:
            y_pred.append(true_ref)
        else:
            y_pred.append(predicted_refs[0] if isinstance(predicted_refs, list) else predicted_refs)

    # Compute accuracy    
    accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)
    
    # Display classifciation report
    print("\nClassification report (references):")
    # TODO check the labels we want (only true or not)
    print(classification_report(y_true, y_pred, labels=list(set(y_true)), zero_division=0))
    report = classification_report(y_true, y_pred, zero_division=0, labels=list(set(y_true)), output_dict=True)
    
    # Extraire les scores pour micro, macro et weighted avg
    metrics_method = ['micro avg', 'macro avg', 'weighted avg']
    metrics = ['precision', 'recall', 'f1-score']
    
    scores = {metric: {method: report[method][metric] for method in metrics_method} for metric in metrics}
    scores['accuracy'] = accuracy

    # Count the number of ref that are not coreectly predicted
    
    nb_wrong_ref = 0
    for instance in predictions:
        if instance['true_reference'] not in instance['predicted_reference']:
            nb_wrong_ref += 1
    print("Number of wrong ref : ", nb_wrong_ref)
    
    # Count the number of classes that are not correctly predicted
    
    nb_wrong_class = 0
    for instance in predictions:
        if instance['true_class'] != instance['predicted_class']:
            nb_wrong_class += 1
    print("Number of wrong class : ", nb_wrong_class)
    
    print("Ratio of wrong class among wrong ref : ", nb_wrong_class/nb_wrong_ref)
    
    
    return scores

def compute_results(scores, method):
    method = 'macro avg'
    
    # Plot the results for the precision base on the evolution of topk
    precision = [scores[i]['precision'][method] for i in range(1,11)]
    accuracy = [scores[i]['accuracy'] for i in range(1,11)]
    recall = [scores[i]['recall'][method] for i in range(1,11)]
    f1 = [scores[i]['f1-score'][method] for i in range(1,11)]
    
    plt.plot(precision, label='Precision')
    plt.plot(accuracy, label='Accuracy')
    plt.plot(recall, label='Recall')
    plt.plot(f1, label='F1-score')
    
    plt.title('Scores evolution with topk for {}'.format(method))
    plt.xlabel('Topk')
    plt.ylabel('Scores')
    
    plt.legend()
    #save the figure
    plt.savefig('res/metrics_results.png')
    

def main(dataset_path, general_model_path, expert_model_dir):
    
    labels = {
            "W Accessories": 0,
            "W Bags": 1,
            "W SLG": 2,
            "W Shoes": 3,
            "Watches": 4,
        }
    
    reverse_labels = {v : k for k,v in labels.items()}

    # Load Test Dataset
    dataset = load_dataset_train_test(dataset_path)
    test_dataset = dataset["test"]
    nb_ref_per_class = Counter(dataset['train']['label'])
    
    # Load the general model
    general_model, processor = load_general_model(general_model_path)

    # Predict the classes for each instance
    class_predictions = predict_general_model(model=general_model, processor=processor, test_dataset=test_dataset, top_k=1)

    # Load the expert models
    expert_model_paths = get_expert_model_paths(expert_model_dir=expert_model_dir, labels=labels)
    expert_models = load_expert_models(expert_model_paths, nb_ref_per_class)       

    scores = {}
    for i in range(10):
        # Predict with the expert models the article's reference
        final_predictions = get_final_predictions(expert_models,class_predictions, test_dataset, reverse_labels, top_k=i+1)
        scores[i+1] = get_ref_scores(final_predictions)
    
    method = 'macro avg'
    compute_results(scores, method)
    


# Si ce fichier est exécuté directement, appeler la fonction main
if __name__ == "__main__":
    
    # TODO change the name if necessary
    
    dataset_path = "augmented_dataset"
    general_model_path = "checkpoint-7350"
    expert_model_dir = "expert_models"
    
    main(dataset_path=dataset_path, general_model_path=general_model_path, expert_model_dir=expert_model_dir)
    