import kagglehub
import shutil
import os
import pandas as pd
def download_data(destination_folder = 'data/raw'):
# Download latest version
    downloaded_path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    # Step 2: Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Step 3: Move the files/folders
    if os.path.isdir(downloaded_path):
        # Move all contents of the folder
        for item in os.listdir(downloaded_path):
            s = os.path.join(downloaded_path, item)
            d = os.path.join(destination_folder, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        # If it's a single file
        shutil.copy2(downloaded_path, destination_folder)
    #shutil.rmtree(downloaded_path)
    
    print(f"Dataset moved to {destination_folder}")
    return destination_folder

def load_elliptic_dataset(data_path):
    # Load features
    features_df = pd.read_csv(data_path+'elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None, 
                              dtype = {0:str}
                            )
    features_df.columns = ['txId', 'time_step']+[f'transaction_{i}' for i in range(93)] + [f'neighbors_{i}' for i in range(72)]
    # Load edges
    edges_df = pd.read_csv(data_path+'elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv', 
                        names=['src', 'dst'], header=0, 
                         dtype = {0:str, 1:str})
    # Load classes
    classes_df = pd.read_csv(data_path+'elliptic_bitcoin_dataset/elliptic_txs_classes.csv', 
                            names=['txId', 'class_label'], 
                            dtype = {0:str}, header=0)
    features_df = features_df.merge(classes_df, on='txId', how='left')
    return features_df, edges_df