import pathlib

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import ViTForImageClassification, ViTImageProcessor

# Paths
root = pathlib.Path(__file__).parent.parent
model_save_path = root / "vit-experiment" / "checkpoint-7350"
dataset_path = root / "augmented_dataset"

# Load the model and tokenizer
model = ViTForImageClassification.from_pretrained(model_save_path)
processor = ViTImageProcessor.from_pretrained(model_save_path)

# Load the test dataset
dataset = load_dataset(
    str(dataset_path),
    name="full_augmented_dataset_with_0_augment",
    num_augments=0,
    trust_remote_code=True,
)
test_dataset = dataset["test"]


# Predict the test dataset
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)


def predict(model: ViTForImageClassification, top_k):
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


metric = load("f1")


def logits(model: ViTForImageClassification, test_dataset):
    model.eval()
    logits = []
    with torch.no_grad():

        inputs = torch.tensor(
            np.array(processor(test_dataset["image"])["pixel_values"]), device=device
        )

    return model.forward(pixel_values=inputs).logits


def evaluate(top_k_predictions, reference, top_k):

    top_k_predictions: torch.Tensor = top_k_predictions[:, :top_k]
    correct_preds = []
    for i in range(len(reference)):
        correct_preds.append(
            reference[i]
            if any(top_k_predictions[i, :] == reference[i])
            else top_k_predictions[i, 0]
        )
    return metric.compute(
        predictions=correct_preds,
        references=reference,
        average="macro",
    )["f1"]


def metrics(logits: torch.Tensor, reference, top_k=2):
    # Return f1 score for 1 to top_k labels
    top_k_predictions = logits.topk(top_k, dim=-1).indices

    return {
        "f1": evaluate(top_k_predictions, reference, 1),
        "top2 f1": evaluate(top_k_predictions, reference, 2),
    }


predictions = logits(model, test_dataset)
reference = test_dataset["label"]
print(metrics(predictions, reference))

exit()
