import pandas as pd
import os
from utils import download_images

def download_images_from_csv(csv_path, download_folder):
    df = pd.read_csv(csv_path)
    image_links = df['image_link']
    
    os.makedirs(download_folder, exist_ok=True)

    print(f"Found {len(image_links)} unique images to download.")

    download_images(image_links, download_folder, allow_multiprocessing=True)
    
    print("Download complete!")

# train_csv_path = '../dataset/train.csv'
# train_download_folder = 'images/train/'

# download_images_from_csv(train_csv_path, train_download_folder)

test_csv_path = '../dataset/test.csv'
test_download_folder = 'images/test/'

download_images_from_csv(test_csv_path, test_download_folder)
