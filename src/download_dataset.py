import pandas as pd
import os
from tqdm import tqdm
from utils import download_images
import random

def download_dataset_images(csv_path, download_folder, num_images=25000):
    df = pd.read_csv(csv_path)
    
    image_links = df['image_link'].unique()

    if len(image_links) < num_images:
        print(f"Only {len(image_links)} unique images found, downloading all.")
        num_images = len(image_links)

    sampled_image_links = random.sample(list(image_links), num_images)

    print(f"Found {len(image_links)} unique images, downloading {num_images} random images.")

    os.makedirs(download_folder, exist_ok=True)

    print("Downloading images...")
    download_images(sampled_image_links, download_folder, allow_multiprocessing=True)

    print("Download complete!")

    downloaded_images = os.listdir(download_folder)
    print(f"Successfully downloaded {len(downloaded_images)} images.")

    missing_images = set(sampled_image_links) - set(downloaded_images)
    if missing_images:
        print(f"Warning: {len(missing_images)} images could not be downloaded.")
        print("First few missing image links:")
        for link in list(missing_images)[:5]:
            print(link)


train_csv_path = '../dataset/train.csv'
train_download_folder = 'images/train/'

download_dataset_images(train_csv_path, train_download_folder, num_images=25000)

test_csv_path = '../dataset/test.csv'
test_download_folder = 'images/test/'

download_dataset_images(test_csv_path, test_download_folder, num_images=10000)
