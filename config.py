# Configuration file for Animal MultiClass Classification

# Hyperparameters
LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)

# Data paths
DATA_DIR = 'afhq/'

# Model parameters
NUM_CLASSES = 3  # cat, dog, wild
MODEL_NAME = 'custom_cnn'

# Device configuration
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Class names
CLASS_NAMES = ['cat', 'dog', 'wild']
