import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

class AnimalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df['labels'])
        self.labels = torch.tensor(self.label_encoder.transform(df['labels']), dtype=torch.long)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {img_path}")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label_encoder(self):
        return self.label_encoder

def create_dataframe(data_dir):
    """Create dataframe from directory structure"""
    image_path = []
    labels = []

    for i in os.listdir(data_dir):
        i_path = os.path.join(data_dir, i)
        if os.path.isdir(i_path):
            for label in os.listdir(i_path):
                label_path = os.path.join(i_path, label)
                if os.path.isdir(label_path):
                    for image in os.listdir(label_path):
                        full_image_path = os.path.join(label_path, image)
                        image_path.append(full_image_path)
                        labels.append(label)

    return pd.DataFrame(zip(image_path, labels), columns=['image_path', 'labels'])

def get_transforms():
    """Get data transforms"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])
    return transform

def split_data(df, train_frac=0.7, val_frac=0.5):
    """Split data into train, validation, and test sets"""
    train = df.sample(frac=train_frac, random_state=42)
    test = df.drop(train.index)
    val = test.sample(frac=val_frac, random_state=42)
    test = test.drop(val.index)

    return train, val, test
