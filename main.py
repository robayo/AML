import os 
from src.data.load_data import download_data
PATH = './data/raw'
if __name__ == '__main__':
    # execute only if run as the entry point into the program
    if not os.path.exists(PATH):
        path = download_data(PATH)
        