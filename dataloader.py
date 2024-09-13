import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ProductImageDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, os.path.basename(self.data.iloc[idx, 1]))
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.data.iloc[idx, 2], self.data.iloc[idx, 3], self.data.iloc[idx, 4]

def maintain_aspect_ratio_resize(image, target_size):
    w, h = image.size
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    return image.resize((new_w, new_h), Image.LANCZOS)

def pad_image(image, target_size):
    w, h = image.size
    left = (target_size - w) // 2
    top = (target_size - h) // 2
    right = target_size - w - left
    bottom = target_size - h - top
    return transforms.functional.pad(image, (left, top, right, bottom), 0, 'constant')

def preprocess_images(image_dir, output_dir, target_size=2048):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess = transforms.Compose([
        transforms.Lambda(lambda img: maintain_aspect_ratio_resize(img, target_size)),
        transforms.Lambda(lambda img: pad_image(img, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                processed_img = preprocess(img)
                
                torch.save(processed_img, os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.pt"))
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

image_dir = 'images/train'
output_dir = 'images/preprocessed/train'
csv_file = '../dataset/train.csv'

preprocess_images(image_dir, output_dir)

dataset = ProductImageDataset(output_dir, csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)