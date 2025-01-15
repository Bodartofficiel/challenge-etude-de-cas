from pathlib import Path

from datasets import load_dataset

dataset_path = Path(__file__).parent.parent / "augmented_dataset"

dataset = load_dataset(str(dataset_path), trust_remote_code=True)
