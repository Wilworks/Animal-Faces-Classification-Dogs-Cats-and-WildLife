# Animal Faces Classification: Dogs, Cats, and Wildlife

This project implements a Convolutional Neural Network (CNN) for multi-class classification of animal faces using PyTorch. The model classifies images into three categories: dogs, cats, and wildlife.

## Dataset

The dataset used is the AFHQ (Animal Faces-HQ) dataset, which contains high-quality images of animal faces. The data is organized in the `afhq/` directory with subdirectories for each class.

## Features

- Custom PyTorch Dataset class for loading and preprocessing images
- CNN architecture with convolutional, pooling, and fully connected layers
- Training with progress bars using tqdm
- Loss and accuracy tracking for training and validation
- Visualization of training progress with matplotlib plots
- Model summary using torchsummary

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- PIL (Pillow)
- matplotlib
- numpy
- pandas
- tqdm
- torchsummary

Install dependencies with: `pip install -r requirements.txt`

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/Wilworks/Animal-Faces-Classification-Dogs-Cats-and-WildLife.git
   cd Animal-Faces-Classification-Dogs-Cats-and-WildLife
   ```

2. Download the AFHQ dataset and place it in the `afhq/` directory.

3. Run the Jupyter notebook `pipeline.ipynb` to train the model.

4. The trained model will be saved as `anim_model.pth`.

## Model Architecture

The CNN consists of:
- 3 convolutional layers with increasing filters (32, 64, 128)
- Max pooling after each conv layer
- ReLU activation
- Fully connected layers: 128*16*16 -> 128 -> 3 (output classes)

Input image size: 128x128x3

## Training

- Optimizer: Adam with learning rate 1e-4
- Loss: Cross-Entropy Loss
- Batch size: 16
- Epochs: 10
- Device: MPS (if available) or CPU

## Results

After training, the notebook generates plots for:
- Training and validation loss over epochs
- Training and validation accuracy over epochs

## License

This project is open-source. Feel free to use and modify.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
