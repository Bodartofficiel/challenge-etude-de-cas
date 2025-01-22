import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

seed = 1
num_augment = 3
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device {device}")

dataset_path = Path(__file__).parent.parent / "augmented_dataset"

dataset = load_dataset(
    str(dataset_path),
    name=f"full_augmented_dataset_with_{num_augment}_augment",
    num_augments=num_augment,
    trust_remote_code=True,
)

label = list(set(dataset["train"]["label"]))
model_name_or_path = "google/vit-base-patch16-224-in21k"

processor = ViTImageProcessor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(label),
    id2label={str(i): c for i, c in enumerate(label)},
    label2id={c: str(i) for i, c in enumerate(label)},
)


def transform(example_batch):
    inputs = processor(
        images=example_batch["image"],
        return_tensors="pt",
        do_resize=True,
        size={"height": 224, "width": 224},
    )
    # Move tensors to the same device as the model
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    inputs["labels"] = torch.tensor(
        example_batch["label"], dtype=torch.long, device=device
    )
    return inputs


prepared_ds = dataset.with_transform(transform)


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])

    # Ensure tensors are contiguous and on the correct device
    pixel_values = pixel_values.contiguous().to(device)
    labels = labels.contiguous().to(device)

    return {"pixel_values": pixel_values, "labels": labels}


metric = load("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


# Update training arguments to handle MPS device
training_args = TrainingArguments(
    output_dir="./vit-experiment-test",
    per_device_train_batch_size=16,  # Reduce batch size if memory issues occur
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch size = 16 * 2 = 32
    num_train_epochs=10,
    learning_rate=5e-5,
    lr_scheduler_type="cosine_with_restarts",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),  # Enable mixed precision for faster training if CUDA is available
    dataloader_pin_memory=False,
    report_to=["mlflow"],  # Disable reporting to external platforms like WandB
)
# Add gradient clipping to prevent potential numerical issues
training_args.max_grad_norm = 1.0

# Initialize trainer with the updated arguments
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    processing_class=processor,
)

# Freeze all layers except the last layer
for name, param in model.named_parameters():
    if "classifier" not in name:  # Ensure you are not freezing the classifier layer
        param.requires_grad = False

train_results = trainer.train()
