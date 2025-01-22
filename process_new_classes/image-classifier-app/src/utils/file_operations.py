import pandas as pd

def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

def save_dataframe_to_csv(dataframe, csv_path):
    dataframe.to_csv(csv_path, index=False)

def load_dataframe_from_csv(csv_path):
    return pd.read_csv(csv_path)