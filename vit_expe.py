import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from PIL import Image
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

# Set seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

model_name_or_path = "google/vit-base-patch16-224-in21k"
# First, let's check and set the device properly
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

ds = load_dataset("imagefolder", data_dir="./cropped-dataset")
label = ds["train"].features["label"]

# Move model to the correct device
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(label),
    id2label={str(i): c for i, c in enumerate(label)},
    label2id={c: str(i) for i, c in enumerate(label)},
).to(device)


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


prepared_ds = ds.with_transform(transform)


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
    output_dir="./vit-experiment",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    num_train_epochs=4,
    lr_scheduler_type="cosine_with_min_lr",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    dataloader_pin_memory=(
        True if device == "cuda" else False
    ),  # Only use pin_memory for CUDA
    use_cpu=True,  # Disable CUDA if not using it
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

# Add a try-except block to catch and print detailed error information
try:
    train_results = trainer.train()
except RuntimeError as e:
    print(f"Detailed error information:")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
    # Print tensor device information
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Sample batch information:")
    sample_batch = next(iter(trainer.get_train_dataloader()))
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape: {v.shape}, device: {v.device}")
