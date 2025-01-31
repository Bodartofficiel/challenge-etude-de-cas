import pathlib
import random

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
    def __init__(self, *args, num_augments, labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_augments = num_augments
        if labels is None:
            self.labels = {
                "W Accessories": 0,
                "W Bags": 1,
                "W SLG": 2,
                "W Shoes": 3,
                "Watches": 4,
            }
        else:
            self.labels = labels

    def _info(self):
        return datasets.DatasetInfo(
            description="Data augmented and cropped images of the dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.Value("int32"),
                    "path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="---",
            citation="---",
        )

    def _split_generators(self, dl_manager):
        data_dir = pathlib.Path(__name__).parent.parent / "data" / "cropped-dataset"
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
                        "path": str(image_file),
                    }
                    i += 1
