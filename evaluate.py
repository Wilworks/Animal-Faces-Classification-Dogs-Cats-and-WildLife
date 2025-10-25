import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import torch.nn as nn

from config import BATCH_SIZE, CLASS_NAMES
from dataset import create_dataframe, split_data, AnimalDataset, get_transforms
from model import AnimalClassifier
from utils import plot_confusion_matrix

def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)

            acc = (torch.argmax(predictions, axis=1) == labels).sum().item()
            total_acc_val += acc

            val_loss = criterion(predictions, labels)
            total_loss_val += val_loss.item()

            all_preds.extend(torch.argmax(predictions, axis=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = total_acc_val / len(val_loader.dataset) * 100
    val_loss = total_loss_val / len(val_loader)

    return val_loss, val_accuracy, all_preds, all_labels

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = create_dataframe('afhq/')
    _, val_df, _ = split_data(df)
    transform = get_transforms()
    val_dataset = AnimalDataset(val_df, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = AnimalClassifier()
    model = model.load_model('animal_classifier.pth')
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    val_loss, val_accuracy, all_preds, all_labels = evaluate_model(model, val_loader, criterion, device)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, "Confusion Matrix - Animal Classification")

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == "__main__":
    main()
