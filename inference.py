import torch
from torchvision import transforms
from PIL import Image
import os

from config import DEVICE, CLASS_NAMES
from model import AnimalClassifier

def preprocess_image(image_path):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_image(model, image_tensor, device):
    """Make prediction on preprocessed image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

def main():
    # Load model
    model = AnimalClassifier()
    model = model.load_model('animal_classifier.pth')

    device = torch.device(DEVICE)
    model.to(device)

    # Example usage
    image_path = input("Enter image path: ")
    if os.path.exists(image_path):
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)

            # Make prediction
            predicted_class, confidence = predict_image(model, image_tensor, device)

            print(f"Predicted class: {CLASS_NAMES[predicted_class]}")
            print(f"Confidence: {confidence:.4f}")

        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("Image path does not exist.")

if __name__ == "__main__":
    main()
