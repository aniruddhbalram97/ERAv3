import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First convolutional block - reduced filters from 8 to 4
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional block - reduced filters from 16 to 8
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third convolutional block - reduced filters from 32 to 16
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2)
        
        # Fully connected layers - reduced hidden layer from 128 to 64
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # First block
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected layers
        x = x.view(-1, 16 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 