import os

import datasets
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

# Load processor and model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
class_tested = 4
top_k = 1

# Load dataset
dataset = load_dataset(
    "./augmented_data/augmented_dataset",
    name="full_augmented_dataset_with_0_augment",
    num_augments=0,
    labels={
        "W Accessories": 0,
        "W Bags": 1,
        "W SLG": 2,
        "W Shoes": 3,
        "Watches": 4,
        "W RTW": 5,
    },
    trust_remote_code=True,
)


# Function to preprocess a batch
# pixel_values is not pixel_values but embedding
def collate(batch):
    return {
        "pixel_values": model(
            **processor(images=batch["image"], return_tensors="pt")
        ).last_hidden_state[:, 0, :]
    }


# Set batch size
batch_size = 16

# Split dataset
train, test = dataset["train"], dataset["test"]
# Dataset({
#     features: ['image', 'label', 'path'],
#     num_rows: 2766
# })

test = test.filter(lambda exemple: exemple["label"] == class_tested)
train = train.filter(lambda example: example["label"] == class_tested)

# Preprocess train dataset

test = test.map(collate, batched=True, batch_size=batch_size, remove_columns="image")


# Directory to store embeddings
os.makedirs("./embeddings", exist_ok=True)

# Step 1: Save train embeddings in chunks
train_embeddings_path = f"embeddings/train_embeddings_0_aug_class_{class_tested}.pt"
print(train_embeddings_path)
if not os.path.exists(train_embeddings_path):

    train = train.map(
        collate, batched=True, batch_size=batch_size, remove_columns="image"
    )
    train_embeddings = torch.tensor(train["pixel_values"])
    torch.save(train_embeddings, train_embeddings_path)
else:
    print("Loading train embeddings from disk...")
    train_embeddings = torch.tensor(torch.load(train_embeddings_path))

# Step 2: Compare test embeddings with train embeddings
print("Computing cosine similarities...")
cosine_similarities = []
correct = 0
total = 0
for batch in tqdm(test, desc="Processing test dataset"):
    test_embeddings = torch.tensor(batch["pixel_values"])
    test_label = batch["path"].split("/")[4][:-6]
    # Compute cosine similarity between test and train embeddings
    similarity = (
        cosine_similarity(test_embeddings, train_embeddings).topk(top_k).indices
    )
    is_in = False
    for i in similarity:
        is_in = is_in or test_label == train[int(i)]["path"].split("/")[4][:-5]

    if is_in:
        correct += 1
    total += 1
    print("\n\n")

    cosine_similarities.append(similarity)
print(correct / total)

# Step 3: Post-process and analyze cosine similarities
cosine_similarities = torch.cat(cosine_similarities)
print("Cosine similarities computed!")
print("Shape of similarity matrix:", cosine_similarities.shape)  # (num_test, num_train)
