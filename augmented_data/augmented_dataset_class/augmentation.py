import random

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms
from torchvision.transforms import (
    ColorJitter,
    GaussianBlur,
    RandomAffine,
    RandomErasing,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    RandomRotation,
)
from transformers import pipeline

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

segment_pipe = pipeline(
    "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True
)


# Définir une fonction personnalisée pour RandomErasing avec une valeur aléatoire
class RandomErasingWithRandomValue:
    def __init__(
        self, p=0.5, scale=(0.05, 0.25), ratio=(0.3, 3.3), base_color=(210, 160, 130)
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.base_color = base_color

    def __call__(self, img):
        if random.random() < self.p:
            # Dimensions de l'image
            _, height, width = img.size()

            # Générer les dimensions d'effacement
            h = random.randint(
                int(self.scale[0] * height), min(int(self.scale[1] * height), height)
            )
            w = random.randint(
                int(self.ratio[0] * width), min(int(self.ratio[1] * width), width)
            )
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)

            # Générer une valeur constante pour la zone (R, G, B)
            factor = random.uniform(
                0.2, 0.9
            )  # Facteur aléatoire pour éclaircir/assombrir
            value = (
                torch.tensor([c * factor for c in self.base_color]) / 255.0
            )  # Normaliser entre 0 et 1
            value = value[:, None, None].expand(
                3, h, w
            )  # Étendre à la taille de la zone effacée

            value = value.expand(3, h, w)  # Étendre à la taille de la zone effacée

            # Appliquer l'effacement
            img[:, i : i + h, j : j + w] = value

        return img


data_augmentation_pipe = transforms.Compose(
    [
        RandomRotation(degrees=90, fill=256),
        RandomHorizontalFlip(p=0.5),
        RandomAffine(degrees=0, translate=(0.2, 0.2), fill=256),
        RandomPerspective(p=0.5, distortion_scale=0.4, fill=256),
        RandomResizedCrop(size=(256, 256), scale=(0.75, 1.5)),
        transforms.ToTensor(),
        GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)),
        RandomErasingWithRandomValue(p=0.5, scale=(0.1, 0.25), ratio=(0.3, 3.3)),
    ]
)


def process_and_augment(image_path, num_augments=20):
    cropped_image  = segment_pipe(image_path).convert("RGB")
    
    augmented_images = []
    for _ in range(num_augments):
        augmented_tensor = data_augmentation_pipe(cropped_image)
        augmented_image = transforms.ToPILImage()(augmented_tensor)
        augmented_images.append(augmented_image)

    return augmented_images
