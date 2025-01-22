import pandas as pd
import os.path as path
import sys,os


SAVE_PATH = "./generated/accessories_classes.csv"
current_dir = os.path.dirname(__file__)
data_path = os.path.join( path.dirname(path.dirname(path.dirname(path.dirname(current_dir)))),"data")

OLD_ID = "MMC"
OLD_CLASSE = "Product_BusinessUnitDesc"


ID = "article_id"
CLASSE = "classe"
FILEPATHS = "filepaths"
SELECTED_CLASS = "selected_class"

def load_csv_to_dataframe(file_path, required_class=None):
    
    try:
        df = pd.read_csv(SAVE_PATH)
        print("data loaded from",SAVE_PATH)
    except FileNotFoundError:
        df = pd.read_csv(file_path)
        df = df.rename(columns={OLD_ID: ID, OLD_CLASSE: CLASSE})
        print("data loaded from",file_path)
        
    if required_class:
        df = df[df[CLASSE] == required_class]

    df[FILEPATHS] = df[ID].apply(lambda x: path.join(data_path, "DAM", x+".jpeg"))
    
    if SELECTED_CLASS not in df.columns:
        df[SELECTED_CLASS] = None
    return df
    
def save_df_to_csv(df, csv_path):
    df[[ID,CLASSE,SELECTED_CLASS]].to_csv(csv_path, index=False)
