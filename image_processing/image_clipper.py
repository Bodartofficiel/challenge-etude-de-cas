import numpy as np
from PIL import Image
from transformers import pipeline


def crop_with_mask_and_resize(image_path):
    # create the pipeline
    pipe = pipeline(
        "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True
    )

    # Charger l'image
    image = Image.open(image_path)

    # Obtenir le masque Pillow
    pillow_mask = pipe(image_path, return_mask=True)
    pillow_image = pipe(image_path)

    # Convertir le masque en NumPy
    mask_array = np.array(pillow_mask)

    # Trouver les indices actifs
    rows, cols = np.where(mask_array > 0)

    # Calculer les limites du masque
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

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
        "RGB", (cropped_image.width, cropped_image.height), (255, 255, 255)
    )

    # Coller l'image recadrée sur le fond blanc
    final_image.paste(cropped_image, (0, 0), cropped_image)

    return final_image.resize((256, 256))


if __name__ == "__main__":

    # Utilisation
    image_path = "data/test_image_headmind/IMG_6875.jpg"  # Chemin de votre image
    cropped_image = crop_with_mask_and_resize(image_path)

    # Afficher ou sauvegarder le résultat
    cropped_image.show()  # Affiche l'image recadrée
