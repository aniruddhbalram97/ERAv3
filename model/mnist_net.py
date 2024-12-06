import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First block
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 