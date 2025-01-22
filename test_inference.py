import torch
from torchvision import transforms
from PIL import Image
import os
from datasets import load_dataset

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

# Load the test dataset
def load_test_set(dataset_path):
    dataset = load_dataset(
        str(dataset_path),
        name="full_augmented_dataset_with_0_augment",
        num_augments=0,
        trust_remote_code=True,
    )
    test_dataset = dataset["test"]
    return test_dataset

# Load the general model
def load_general_model(model_path):
    model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()
    processor = ViTImageProcessor.from_pretrained(model_path)
    # model.eval()
    return model, processor

# TODO Load the expert models
def load_expert_models(expert_model_paths):
    expert_models = {}
    for i,path in enumerate(expert_model_paths):
        expert_models[i] = ViTForImageClassification.from_pretrained(path).to(device).eval()
    return expert_models

# Predict the classes
def predict_general_model(model: ViTForImageClassification, processor, test_dataset, top_k):
    model.eval()
    predictions = []
    with torch.no_grad():
        for image in test_dataset["image"]:

            inputs = torch.tensor(
                np.array(processor(image)["pixel_values"]), device=device
            )
            outputs = model.forward(pixel_values=inputs)
            logits = outputs.logits
            top2_indices = torch.topk(logits, top_k).indices.cpu().numpy().tolist()[0]
            predictions.append(top2_indices)
    return predictions


# Predict the articles
def get_final_predictions(expert_models, class_predictions, test_dataset):
    
    # Predict with the expert models
    final_predictions = []
    
    for idx, class_pred in enumerate(class_predictions):
        # Get the predicted class (first element of the prediction)
        predicted_class = class_pred[0]
        
        # Get the corresponding expert model
        expert_model = expert_models[predicted_class]
        
        # Get the image from the dataset
        image = test_dataset[idx]
        
        # Predict the reference using the expert model
        reference_prediction = predict_expert_model(
            model=expert_model, image = image
        )
        
        # TODO add the true class and true reference
        final_predictions.append({
            'image_idx': idx,
            'predicted_class': predicted_class,
            'predicted_reference': reference_prediction
        })
        
    return final_predictions


# Predict the article's reference
def predict_expert_model(model, image):
    pass


# Functions to get all the paths for 
def get_expert_model_paths(expert_model_dir):
    expert_model_paths = []
    for chkpt in os.listdir(expert_model_dir):
        chkpt_path = os.path.join(expert_model_dir, chkpt)
        expert_model_paths.append(chkpt_path)
    return expert_model_paths

# Functions to compute the score and store graphs 
def compute_results():
    pass


def main(dataset_path, general_model_path, expert_model_dir):
    
    # Load Test Dataset
    test_dataset = load_test_set(dataset_path)
    
    # Load the general model
    general_model, processor = load_general_model(general_model_path)

    # Predict the classes for each instance
    class_predictions = predict_general_model(model=general_model, processor=processor, test_dataset=test_dataset, top_k=1)
    print(class_predictions)

    
    # Load the expert models
    expert_model_paths = get_expert_model_paths(expert_model_dir)
    expert_models = load_expert_models(expert_model_paths)       
    
    
    # Predict with the expert models the article's reference
    final_predictions = get_final_predictions(expert_models, class_predictions, test_dataset)
    
    # Store evaluation results on graphs
    compute_results(final_predictions)
    
    return final_predictions



# Si ce fichier est exécuté directement, appeler la fonction main
if __name__ == "__main__":
    dataset_path = "augmented_dataset"
    general_model_path = "checkpoint-7350"
    expert_model_dir = ""
    
    main(dataset_path=dataset_path, general_model_path=general_model_path, expert_model_dir=expert_model_dir)
    