import matplotlib.pyplot as plt
import numpy as np
import torch
from config import CLASS_NAMES

def plot_class_distribution(df, title="Class Distribution"):
    """Plot class distribution bar chart"""
    df['labels'].value_counts().plot.bar()
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.show()

def plot_training_history(loss_train, loss_val, acc_train, acc_val, title="Training History"):
    """Plot training loss and accuracy"""
    epochs = range(1, len(loss_train) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train, label='Training Loss')
    plt.plot(epochs, loss_val, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_train, label='Training Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

def show_sample_images(df, n_rows=3, n_cols=3):
    """Display sample images from dataframe"""
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = np.random.randint(0, len(df))
            image_path = df.iloc[idx]['image_path']
            image = plt.imread(image_path)
            axarr[i, j].imshow(image)
            axarr[i, j].set_title(f'Class: {df.iloc[idx]["labels"]}')
            axarr[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn"""
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
