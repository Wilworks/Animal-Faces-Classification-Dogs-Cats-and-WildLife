import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from config import LR, EPOCHS, BATCH_SIZE, DATA_DIR
from dataset import create_dataframe, split_data, AnimalDataset, get_transforms
from model import AnimalClassifier
from utils import plot_training_history

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Training loop"""
    model.train()
    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    for epoch in range(num_epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_acc_val = 0
        total_loss_val = 0

        # Training
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training')
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            total_loss_train += train_loss.item()
            train_loss.backward()
            optimizer.step()

            train_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
            total_acc_train += train_acc
            train_progress.set_postfix(loss=train_loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Validation')
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_loss_val += val_loss.item()

                val_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
                total_acc_val += val_acc

        # Store metrics
        total_loss_train_plot.append(round(total_loss_train / len(train_loader), 4))
        total_loss_validation_plot.append(round(total_loss_val / len(val_loader), 4))
        total_acc_train_plot.append(round(total_acc_train / len(train_loader.dataset) * 100, 4))
        total_acc_validation_plot.append(round(total_acc_val / len(val_loader.dataset) * 100, 4))

        print(f'''Epoch: {epoch+1}/{num_epochs}, Train Loss: {total_loss_train_plot[-1]}, Train Accuracy: {total_acc_train_plot[-1]}%
Validation Loss: {total_loss_validation_plot[-1]}, Validation Accuracy: {total_acc_validation_plot[-1]}%
''')

    return total_loss_train_plot, total_loss_validation_plot, total_acc_train_plot, total_acc_validation_plot

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split data
    df = create_dataframe(DATA_DIR)
    train_df, val_df, test_df = split_data(df)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Create datasets
    transform = get_transforms()
    train_dataset = AnimalDataset(train_df, transform=transform)
    val_dataset = AnimalDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = AnimalClassifier()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Train model
    loss_train, loss_val, acc_train, acc_val = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS, device
    )

    # Plot training history
    plot_training_history(loss_train, loss_val, acc_train, acc_val, "Animal Classification")

    # Save model
    model.save_model('animal_classifier.pth')
    print("Model saved as 'animal_classifier.pth'")

if __name__ == "__main__":
    main()
