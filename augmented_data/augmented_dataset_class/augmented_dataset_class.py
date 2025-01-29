import pathlib
import random
import os
import cv2
import datasets
import numpy as np
import PIL
import PIL.Image
import torch

from .augmentation import process_and_augment

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class CustomDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, *args, num_augments, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_augments = num_augments
        self.labels = self.create_labels_dict("cropped-dataset-class/articles_train")

    def _info(self):
        return datasets.DatasetInfo(
            description="Data augmented and cropped images of the dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
            homepage="---",
            citation="---",
        )

    def _split_generators(self, dl_manager):
        data_dir = pathlib.Path(__name__).parent.parent / "data/cropped-dataset-class"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir / "articles_train", "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir / "articles_test", "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: pathlib.Path, split):
        if split == "train":
            augment_func = lambda x: process_and_augment(x, self.num_augments)
        elif split == "test":
            augment_func = lambda x: [x]
        else:
            raise ValueError(f"Unknown name for split: {split}")
        i = 0
        for label, label_idx in self.labels.items():
            for image_file in (filepath / label).glob("*"):

                for image in augment_func(image_file):
                    yield i, {
                        "image": image,
                        "label": label_idx,
                    }
                    i += 1

    def create_labels_dict(self, base_path):
        # Get all subfolders (classes) in the base path
        classes = sorted([folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))])
        
        # Create labels_dict mapping folder names to unique integers
        labels_dict = {class_name: idx for idx, class_name in enumerate(classes)}
        return labels_dict
