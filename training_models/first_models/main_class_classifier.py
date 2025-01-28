import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
    MobileNetV2ForImageClassification,
    MobileNetV2ImageProcessor
)
from pathlib import Path

class PipelineTrain:
    def __init__(self, model_name, dataset_path, labels_dict):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.labels_dict = labels_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = self.use_processor()

    def use_processor(self):
        if self.model_name == "google/vit-base-patch16-224-in21k":
            return ViTImageProcessor.from_pretrained(self.model_name)
        elif self.model_name == "google/mobilenet_v2_1.0_224":
            return  MobileNetV2ImageProcessor.from_pretrained(self.model_name)

    def transform(self, example_batch):
        inputs = self.processor(
            images=example_batch["image"],
            return_tensors="pt",
            size={"height": 224, "width": 224},
        )
        inputs["labels"] = torch.tensor(example_batch["label"], dtype=torch.long)
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["labels"] for x in batch])
        return {"pixel_values": pixel_values, "labels": labels}
    
    def create_augmented_dataset(self):
        return load_dataset(str(self.dataset_path), trust_remote_code=True, num_augments=3)
    
    def prepare_model(self):
        self.id2label = {str(idx): label for label, idx in self.labels_dict.items()}
        self.label2id = {label: str(idx) for label, idx in self.labels_dict.items()}
        if self.model_name == "google/vit-base-patch16-224-in21k":
            self.model = ViTForImageClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.labels_dict),
                id2label=self.id2label,
                label2id=self.label2id,
            ).to(self.device)
        elif self.model_name == "google/mobilenet_v2_1.0_224":
            self.model = MobileNetV2ForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels_dict),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
            ).to(self.device)
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2),  
                torch.nn.Linear(in_features=1280, out_features=len(self.labels_dict)) 
            )

    def compute_metrics(self, p):
        predictions = np.argmax(p.predictions, axis=1)
        metric = load("f1")
        return metric.compute(predictions=predictions, references=p.label_ids)
    
    def initialize_pipeline(self):
        # Load augmented dataset
        ds = self.create_augmented_dataset()
        # Prepare model
        self.prepare_model()
        # Preprocess dataset
        prepared_ds = ds.with_transform(self.transform)
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./mobilenetv2-experiment",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=4,
            learning_rate=5e-5,
            lr_scheduler_type="cosine_with_restarts",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            report_to=["none"],
        )
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn,
            train_dataset=prepared_ds["train"], 
            eval_dataset=prepared_ds["test"],
            compute_metrics=self.compute_metrics,
        )
        return trainer


# Train the model
try:
    # Initialize pipeline
    pipeline = PipelineTrain(model_name = "google/mobilenet_v2_1.0_224",
                    dataset_path = Path(__file__).parent.parent / "augmented_dataset" , 
                    labels_dict = {"W Accessories": 0,"W Bags": 1,"W SLG": 2,"W Shoes": 3,"Watches": 4})
    trainer = pipeline.initialize_pipeline()
    train_results = trainer.train()
    print("Training complete.")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
    print("Model device:", next(pipeline.model.parameters()).device)
    sample_batch = next(iter(trainer.get_train_dataloader()))
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, device={v.device}")
