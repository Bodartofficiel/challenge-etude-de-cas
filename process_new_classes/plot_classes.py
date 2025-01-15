import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os.path as path


# ['W RTW' 'W SLG' 'W Bags' 'W Shoes' 'Watches' 'W Accessories']


# classe = rd.sample(list(classes),1)[0]

def show_group(image_filepaths):

    # Définir le nombre de colonnes pour la grille
    num_images = len(image_filepaths)
    print("Nombre d'images :", num_images)

    if num_images == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        img = mpimg.imread(image_filepaths[0])
        ax.imshow(img)
        ax.axis('off')
        
    elif num_images<=2 :
        num_cols = 2
        num_rows = 1
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        for i, file in enumerate(image_filepaths):
            img = mpimg.imread(file)
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')


    else:
        
        num_cols = np.floor(np.sqrt(num_images)).astype(int)
        if num_cols**2 < num_images:
            num_cols += 1
        num_rows = num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        for i, file in enumerate(image_filepaths):
            try:
                img = mpimg.imread(file)
                ax = axes[i // num_cols, i % num_cols]
                ax.imshow(img)
                ax.axis('off')
            except:
                print("Error with file", file)

    plt.tight_layout()
    plt.show()