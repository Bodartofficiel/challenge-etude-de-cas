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
        data_dir = pathlib.Path(__name__).parent.parent / "cropped-dataset"
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
        labels_to_dict = {
            str(label).split("/")[-1]: i for i, label in enumerate(filepath.glob("*"))
        }
        if split == "train":
            augment_func = process_and_augment
        elif split == "test":
            augment_func = lambda x, y: [x]
        else:
            raise ValueError(f"Unknown name for split: {split}")

        i = 0
        for label, label_idx in labels_to_dict.items():
            for image_file in (filepath / label).glob("*"):
                image_pil = PIL.Image.open(image_file)
                image_np_rgb = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

                for image in augment_func(image_pil, image_np_rgb):
                    yield i, {
                        "image": image,
                        "label": label_idx,
                    }
                    i += 1
