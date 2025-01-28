import matplotlib.image as mpimg
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import pandas as pd
import glob
from tqdm import tqdm
import math


# ['W RTW' 'W SLG' 'W Bags' 'W Shoes' 'Watches' 'W Accessories']


# classe = rd.sample(list(classes),1)[0]

def show_group(image_filepaths):

    # Définir le nombre de colonnes pour la grille
    num_images = len(image_filepaths)
    print("Nombre d'images :", num_images)

    if num_images == 1:
        fig, ax = plt.subplots(figsize=(2, 2))
        img = mpimg.imread(image_filepaths[0])
        ax.imshow(img)
        ax.set_title(image_filepaths[0].strip(".jpeg").split('/')[-1])
        ax.axis('off')
        
    else:
        num_cols = np.floor(np.sqrt(num_images)).astype(int)
        
        if num_cols**2 < num_images:
            num_cols += 1
        num_rows = num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_cols*2))

        for i, file in enumerate(image_filepaths):
            try:
                img = mpimg.imread(file)
                ax = axes.flatten()[i]
                ax.imshow(img)
                ax.set_title(file.strip(".jpeg").split('/')[-1])
                ax.axis('off')
            except:
                print("Error with file", file)
                
        for i in range(num_images, num_rows*num_cols):
            fig.delaxes(axes.flatten()[i])
        
    # elif num_images<=2 :
    #     num_cols = 2
    #     num_rows = 1
    #     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    #     for i, file in enumerate(image_filepaths):
    #         img = mpimg.imread(file)
    #         ax = axes[i]
    #         ax.imshow(img)
    #         ax.set_title(file.strip(".jpeg").split('/')[-1])
    #         ax.axis('off')


    # else:
        
    #     num_cols = np.floor(np.sqrt(num_images)).astype(int)
    #     if num_cols**2 < num_images:
    #         num_cols += 1
    #     num_rows = num_cols

    #     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    #     for i, file in enumerate(image_filepaths):
    #         try:
    #             img = mpimg.imread(file)
    #             ax = axes[i // num_cols, i % num_cols]
    #             ax.imshow(img)
    #             ax.set_title(file.strip(".jpeg").split('/')[-1])
    #             ax.axis('off')
    #         except:
    #             print("Error with file", file)
    #     for i in range(num_images, num_rows*num_cols):
    #         fig.delaxes(axes.flatten()[i])
    

    plt.tight_layout()
    plt.show()
    
def plot_classes(df:pd.DataFrame, column_class="class", column_path="path", column_id = "article_id"):

    classes_count = df.groupby(column_class)[column_id].count()
    print(classes_count.index)
    print(classes_count.iloc[70:77])
    column_number = classes_count.max()
    line_number = len(classes_count)
    
    print("number of classes:", line_number)
    print("max number of images per class:", column_number)
    
    fig, axes = plt.subplots(line_number, column_number, figsize=(column_number//2, line_number//2))
    for i, cla in tqdm(enumerate(classes_count.index)):
        class_df = df[df[column_class] == cla]
        for j in range(len(class_df)):
            img = mpimg.imread(class_df.iloc[j][column_path])
            ax:Axes = axes[i, j]
            ax.imshow(img)
            ax.set_title(class_df.iloc[j][column_id],{'fontsize': 2})
            ax.axis('off')
        for j in range(len(class_df), column_number):
            fig.delaxes(axes[i, j])

    plt.tight_layout()
    plt.show()
    
def plot_classes2(df: pd.DataFrame, column_class="class", column_path="path", column_id="article_id"):
    classes_count = df.groupby(column_class)[column_id].count()
    print(classes_count.index)
    print(classes_count.iloc[70:77])
    
    fig = plt.figure(figsize=(20, 20))
    colors = plt.cm.get_cmap('tab20', len(classes_count))  # Utiliser une colormap pour les couleurs de fond
    
    outer_grid = fig.add_gridspec(len(classes_count), 1, wspace=0, hspace=0)
    
    for i, cla in tqdm(enumerate(classes_count.index)):
        class_df = df[df[column_class] == cla]
        num_images = len(class_df)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        
        inner_grid = outer_grid[i].subgridspec(rows, cols, wspace=0.1, hspace=0.1)
        for j in range(num_images):
            img = mpimg.imread(class_df.iloc[j][column_path])
            ax = fig.add_subplot(inner_grid[j // cols, j % cols])
            ax.imshow(img)
            ax.set_title(class_df.iloc[j][column_id])
            ax.axis('off')
            ax.set_facecolor(colors(i))  # Définir la couleur de fond pour chaque subplot
    
    plt.tight_layout()
    plt.show()
    
    
    
def load_csv_to_dataframe(csv_filepath):
    print(f"Loading CSV file from '{csv_filepath}'")
    return pd.read_csv(csv_filepath)
    


if __name__ == "__main__":
    
    data_path = "../data/data/"
    train_dirname = "DAM"
    
    csv_filepath = "./classes/n796/*.csv"
    possible_csv_files = glob.glob(csv_filepath)
    if possible_csv_files:
        csv_filepath = possible_csv_files[0]
        print(f"Found CSV file: {csv_filepath}")
    else:
        raise FileNotFoundError(f"No CSV files found matching the pattern {csv_filepath}")
    

    infos = load_csv_to_dataframe(csv_filepath)
    infos["filepaths"] = infos["article_id"].apply(lambda x: path.join(data_path, train_dirname, x+".jpeg"))
    print("columns:",infos.columns)
    
    print("loading bags")
    Bags = infos[infos["categorie"]=="W Bags"]
    # print(Bags.sample(5))
    plot_classes(Bags, column_class="classe", column_path="filepaths", column_id = "article_id")
    

    # classes = infos["classe"].unique()
    # print("classes : ",classes)
    # print(infos.columns)    
    # print("\nCOUNT:")
    # print(infos.groupby("classe").count())
    # df = pd.read_csv("data/df_articles.csv")
    # plot_classes(df)