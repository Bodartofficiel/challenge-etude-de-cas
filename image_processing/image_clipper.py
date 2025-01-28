"""Run this file alone to generate the test and train images with removed background"""

import pathlib

import numpy as np
from PIL import Image
from transformers import pipeline

pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)


def crop_with_mask_and_resize(image_path):
    # create the pipeline

    # Charger l'image
    image = Image.open(image_path)

    # Obtenir le masque Pillow
    pillow_mask = pipe(image_path, return_mask=True)
    pillow_image = pipe(image_path)

    # Convertir le masque en NumPy
    mask_array = np.array(pillow_mask)

    # Trouver les indices actifs
    rows, cols = np.sum(mask_array, axis=1), np.sum(mask_array, axis=0)

    rows, cols = np.where(rows > len(cols) / 10)[0], np.where(cols > len(rows) / 10)[0]

    # Trouver la séquence la plus longue d'indices non interrompue
    def longest_non_interrupted_sequence(indices):
        max_sublist = [indices[0]]
        sublist = [indices[0]]
        for i in range(1, len(indices)):
            if int(indices[i]) == int(indices[i - 1]) + 1:
                sublist.append(int(indices[i]))
            else:
                if len(sublist) > len(max_sublist):
                    max_sublist = sublist
                sublist = [int(indices[i])]
        if len(sublist) > len(max_sublist):
            max_sublist = sublist

        return max_sublist

    longest_rows_sequence = longest_non_interrupted_sequence(rows)
    longest_cols_sequence = longest_non_interrupted_sequence(cols)

    # Calculer les limites du masque
    min_row, max_row = longest_rows_sequence[0], longest_rows_sequence[-1]
    min_col, max_col = longest_cols_sequence[0], longest_cols_sequence[-1]

    # Calculer la largeur et la hauteur
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # Déterminer la dimension maximale (largeur ou hauteur)
    if height > width:
        # Ajuster la largeur pour centrer
        pad = (height - width) // 2
        min_col = max(0, min_col - pad)
        max_col = min(image.width, max_col + pad)
    elif width > height:
        # Ajuster la hauteur pour centrer
        pad = (width - height) // 2
        min_row = max(0, min_row - pad)
        max_row = min(image.height, max_row + pad)

    # Recadrer l'image
    cropped_image = pillow_image.crop((min_col, min_row, max_col, max_row))
    # Créer une nouvelle image avec un fond blanc
    final_image = Image.new(
        "RGB",
        (
            max(cropped_image.width, cropped_image.height),
            max(cropped_image.width, cropped_image.height),
        ),
        (255, 255, 255),
    )

    # Coller l'image recadrée sur le fond blanc
    if cropped_image.width > cropped_image.height:
        h = (final_image.height - cropped_image.height) // 2
        w = 0
    else:
        w = (final_image.width - cropped_image.width) // 2
        h = 0

    final_image.paste(cropped_image, (w, h), cropped_image)

    return final_image.resize((256, 256))


if __name__ == "__main__":

    # Utilisation
    file_path = pathlib.Path(__file__).resolve()
    main_dir = file_path.parent.parent
    dir = "test_image_headmind"
    image_dir_path = main_dir / "data/data" / dir
    image_dir = list(image_dir_path.glob("*"))

    # image_path = "data/test_image_headmind/IMG_6934.jpg"
    # crop_with_mask_and_resize(image_path).show()

    for img_path in reversed(image_dir):
        image_path = str(img_path)
        cropped_image = crop_with_mask_and_resize(image_path)

        # Afficher ou sauvegarder le résultat
        output_dir = main_dir / "data/data-cropped" / dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / img_path.name
        cropped_image.save(output_path)
        # cropped_image.show(title=image_path.split("/")[-1])  # Affiche l'image recadrée

    # Utilisation
    file_path = pathlib.Path(__file__).resolve()
    main_dir = file_path.parent.parent
    dir = "DAM"
    image_dir_path = main_dir / "data/data" / dir
    image_dir = list(image_dir_path.glob("*"))

    # image_path = "data/test_image_headmind/IMG_6934.jpg"
    # crop_with_mask_and_resize(image_path).show()

    for img_path in reversed(image_dir):
        image_path = str(img_path)
        cropped_image = crop_with_mask_and_resize(image_path)

        # Afficher ou sauvegarder le résultat
        output_dir = main_dir / "data/data-cropped" / dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / img_path.name
        cropped_image.save(output_path)
        # cropped_image.show(title=image_path.split("/")[-1])  # Affiche l'image recadrée
