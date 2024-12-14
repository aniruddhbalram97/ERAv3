# MNIST Digit Recognizer using CNN

## Description
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The model is designed to achieve high accuracy while maintaining a low parameter count, making it efficient for training and inference. The architecture includes convolutional layers, batch normalization, dropout for regularization, and data augmentation techniques to improve model robustness.

## Build and Run

### Prerequisites
- Python 3.8 or higher
- PyTorch
- torchvision
- pytest

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist_digit_recognizer.git
   cd mnist_digit_recognizer
   ```

2. Install the required packages:
   ```bash
   pip install -e .
   ```

### Training the Model
To train the model, run the following command:
```bash
python train.py
```
This will start the training process, and the model will be trained on the MNIST dataset. Training logs will be printed to the console, showing the training and test accuracy for each epoch.

### Running Tests Locally
To run the tests locally, use the following command:
To test the model, run the following command:
```bash
python test.py
```
This will execute the test suite and verify that the model meets the specified criteria, including parameter count, use of batch normalization, dropout, and fully connected layers.

## Training Logs
During training, the following logs will be displayed for each epoch:

Epoch [1/20]
Train Loss: 0.7830, Train Accuracy: 81.56%
Test Loss: 0.1158, Test Accuracy: 97.20%
New best model saved with accuracy: 97.20%
------------------------------------------------------------
Epoch [2/20]
Train Loss: 0.1055, Train Accuracy: 97.00%
Test Loss: 0.0552, Test Accuracy: 98.43%
New best model saved with accuracy: 98.43%
------------------------------------------------------------
Epoch [3/20]
Train Loss: 0.0663, Train Accuracy: 97.99%
Test Loss: 0.0529, Test Accuracy: 98.27%
------------------------------------------------------------
Epoch [4/20]
Train Loss: 0.0535, Train Accuracy: 98.33%
Test Loss: 0.0563, Test Accuracy: 98.04%
------------------------------------------------------------
Epoch [5/20]
Train Loss: 0.0477, Train Accuracy: 98.51%
Test Loss: 0.0336, Test Accuracy: 98.89%
New best model saved with accuracy: 98.89%
------------------------------------------------------------
Epoch [6/20]
Train Loss: 0.0417, Train Accuracy: 98.68%
Test Loss: 0.0306, Test Accuracy: 98.97%
New best model saved with accuracy: 98.97%
------------------------------------------------------------
Epoch [7/20]
Train Loss: 0.0347, Train Accuracy: 98.91%
Test Loss: 0.0311, Test Accuracy: 98.88%
------------------------------------------------------------
Epoch [8/20]
Train Loss: 0.0310, Train Accuracy: 99.02%
Test Loss: 0.0377, Test Accuracy: 98.89%
------------------------------------------------------------
Epoch [9/20]
Train Loss: 0.0243, Train Accuracy: 99.20%
Test Loss: 0.0425, Test Accuracy: 98.64%
------------------------------------------------------------
Epoch [10/20]
Train Loss: 0.0226, Train Accuracy: 99.27%
Test Loss: 0.0367, Test Accuracy: 98.84%
------------------------------------------------------------
Epoch [11/20]
Train Loss: 0.0189, Train Accuracy: 99.41%
Test Loss: 0.0278, Test Accuracy: 99.24%
New best model saved with accuracy: 99.24%
------------------------------------------------------------
Epoch [12/20]
Train Loss: 0.0156, Train Accuracy: 99.50%
Test Loss: 0.0265, Test Accuracy: 99.17%
------------------------------------------------------------
Epoch [13/20]
Train Loss: 0.0114, Train Accuracy: 99.64%
Test Loss: 0.0316, Test Accuracy: 99.11%
------------------------------------------------------------
Epoch [14/20]
Train Loss: 0.0087, Train Accuracy: 99.74%
Test Loss: 0.0246, Test Accuracy: 99.33%
New best model saved with accuracy: 99.33%
------------------------------------------------------------
Epoch [15/20]
Train Loss: 0.0062, Train Accuracy: 99.80%
Test Loss: 0.0231, Test Accuracy: 99.36%
New best model saved with accuracy: 99.36%
------------------------------------------------------------
Epoch [16/20]
Train Loss: 0.0041, Train Accuracy: 99.88%
Test Loss: 0.0257, Test Accuracy: 99.29%
------------------------------------------------------------
Epoch [17/20]
Train Loss: 0.0031, Train Accuracy: 99.92%
Test Loss: 0.0258, Test Accuracy: 99.32%
------------------------------------------------------------
Epoch [18/20]
Train Loss: 0.0023, Train Accuracy: 99.93%
Test Loss: 0.0248, Test Accuracy: 99.38%
New best model saved with accuracy: 99.38%
------------------------------------------------------------
Epoch [19/20]
Train Loss: 0.0019, Train Accuracy: 99.95%
Test Loss: 0.0244, Test Accuracy: 99.40%
New best model saved with accuracy: 99.40%
------------------------------------------------------------
Epoch [20/20]
Train Loss: 0.0014, Train Accuracy: 99.98%
Test Loss: 0.0244, Test Accuracy: 99.39%
------------------------------------------------------------
Best Test Accuracy: 99.40%