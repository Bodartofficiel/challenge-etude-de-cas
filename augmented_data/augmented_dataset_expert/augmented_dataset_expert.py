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

from datasets import BuilderConfig

class CustomDatasetConfig(BuilderConfig):
    def __init__(self, num_augments=1, class_name = "W Bags", **kwargs):
        super().__init__(**kwargs)
        self.num_augments = num_augments
        self.class_name = class_name

class CustomDataset(datasets.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [
        CustomDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default configuration with custom augmentation",
            num_augments=1,
            class_name = "W Bags"
            
        )
    ]
    def _info(self):
        return datasets.DatasetInfo(
            description="Data augmented and cropped images of the dataset",
            features=datasets.Features(
            {
                "image": datasets.Image(),
                "label": datasets.Value("string"),
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
                gen_kwargs={"filepath": data_dir / "articles_train", "split": "train",
                "class_name": self.config.class_name, "num_augments": self.config.num_augments,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir / "articles_test", "split": "test",     
                 "class_name": self.config.class_name, "num_augments": self.config.num_augments,
                 },

            ),
        ]

    def _generate_examples(self, filepath: pathlib.Path, split, class_name, num_augments):
        
        if split == "train":
            augment_func = process_and_augment
        elif split == "test":
            augment_func = lambda x, y, z: [x]
        else:
            raise ValueError(f"Unknown name for split: {split}")

        i = 0

        for image_file in (filepath / class_name).glob("*"):
                label = str(image_file.name).replace("_0", "")  # Convertir en chaîne et remplacer
                for x in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    label = label.replace("_" + str(x), "")  # Convertir en chaîne et remplacer
                label = label.replace(".jpg", "")
                label = label.replace(".jpeg", "")
                image_pil = PIL.Image.open(image_file)
                image_np_rgb = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

                for image in augment_func(image_pil, image_np_rgb, num_augments):
                    yield i, {
                        "image": image,
                        "label": label,
                    }
                    i += 1
