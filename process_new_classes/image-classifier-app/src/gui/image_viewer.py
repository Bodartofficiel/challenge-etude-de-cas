import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
from gui.data_handler import load_csv_to_dataframe, save_df_to_csv


SAVE_DIR = "./generated"
SAVE_NAME = "accessories_classes.csv"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

class ImageViewer:
    def __init__(self, master, classes, dataframe):
        self.master = master
        self.classes = classes
        self.dataframe: pd.DataFrame = dataframe
        
        # show remaining images
        self.remaining_indexes = self.dataframe.loc[self.dataframe['selected_class'].isnull()].index.tolist() 
        for aba in ["autres","bijoux"]:
            self.remaining_indexes += self.dataframe.loc[self.dataframe['selected_class'] == aba].index.tolist()
        print(self.remaining_indexes)
        self.remaining_images_label = tk.Label(master, text=f"Remaining images: {len(self.remaining_indexes)}")
        self.remaining_images_label.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.class_var = tk.StringVar(value=classes[0])
        self.class_menu = tk.OptionMenu(master, self.class_var, *classes)
        self.class_menu.pack()

        self.next_button = tk.Button(master, text="Next", command=self.next_image)
        self.next_button.pack()
        
    def display_image(self):
        image_path = self.dataframe.loc[self.remaining_indexes[0]]['filepaths']
        image = Image.open(image_path)
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        selected = self.dataframe.loc[self.remaining_indexes[0]]['selected_class']
        if selected in self.classes:
            self.class_var.set(selected)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.remaining_images_label.config(text=f"Remaining images: {len(self.remaining_indexes)}")

    def next_image(self):
        selected_class = self.class_var.get()
        article_id = self.dataframe.loc[self.remaining_indexes[0]]['article_id']
        print(article_id,":",selected_class)
        self.dataframe.loc[self.remaining_indexes[0],"selected_class"] = selected_class
        save_df_to_csv(self.dataframe, SAVE_PATH)
        self.remaining_indexes.pop(0)
        if len(self.remaining_indexes) == 0:
            messagebox.showinfo("Info", "All images have been classified")
            self.master.quit()
            return
        self.display_image()
        