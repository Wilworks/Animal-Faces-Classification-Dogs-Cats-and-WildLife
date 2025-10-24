import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Define the model architecture (same as in training)
class Anim(nn.Module):
    def __init__(self, num_classes=3):
        super(Anim, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = Anim(num_classes=3)
model.load_state_dict(torch.load('anim_model.pth', map_location=device))
model.to(device)
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

# Load label encoder (assuming it's saved or recreated)
# For simplicity, recreate with known classes
label_encoder = LabelEncoder()
label_encoder.fit(['cat', 'dog', 'wild'])  # Adjust based on your classes

def predict_image(image_path):
    """
    Predict the class of a single image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Predicted class label.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = label_encoder.inverse_transform([predicted.item()])[0]

        return predicted_class
    except Exception as e:
        return f"Error predicting image: {str(e)}"

def predict_batch(image_paths):
    """
    Predict classes for a batch of images.

    Args:
        image_paths (list): List of paths to image files.

    Returns:
        list: List of predicted class labels.
    """
    predictions = []
    for path in image_paths:
        pred = predict_image(path)
        predictions.append(pred)
    return predictions

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter the path to an image: ")
    if os.path.exists(image_path):
        prediction = predict_image(image_path)
        print(f"Predicted class: {prediction}")
    else:
        print("Image path does not exist.")

    # For batch prediction, uncomment and modify:
    # image_paths = ["path1.jpg", "path2.jpg"]
    # predictions = predict_batch(image_paths)
    # print(predictions)
