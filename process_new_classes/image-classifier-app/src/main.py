import tkinter as tk
from tkinter import messagebox
from gui.image_viewer import ImageViewer
from gui.data_handler import load_csv_to_dataframe, save_df_to_csv

CLASS_NAME = "W Accessories"
FILE_NAME = "accessories_classes.csv"



CLASSES =[
    "foulard",
    "bandeau",
    "bijoux",
    "bagues",
    "serre-tete",
    "collier",
    "ance",
    "lunettes",
    "casquette",
    "bob",
    "bracelet",
    "ceinture",
    "pareo",
    "echarpe",
    "chaussettes",
    "gants",
    "tourdecou",
    "autres"
]

class ImageClassifierApp:
    def __init__(self, master, classes,dataframe_path):
        self.master = master
        self.master.title("Image Classifier")
        self.dataset = load_csv_to_dataframe(dataframe_path,CLASS_NAME)
        self.current_index = 0
        self.image_viewer = ImageViewer(master, classes, self.dataset)

        self.show_next_image()

    def show_next_image(self):
        self.image_viewer.display_image()
        
    def save_dataset(self):
        save_df_to_csv(self.dataset, FILE_NAME)

    # def record_selection(self, selected_class):
    #     self.dataset.at[self.current_index, 'selected_class'] = selected_class
    #     self.dataset.to_csv(FILE_NAME, index=False)
    #     self.current_index += 1
    #     self.show_next_image()

if __name__ == "__main__":
    root = tk.Tk()
    dataframe_path = "../../data/product_list.csv"
    app = ImageClassifierApp(root, CLASSES, dataframe_path)
    root.mainloop()