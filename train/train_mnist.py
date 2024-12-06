import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.mnist_net import MNISTNet

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.002

    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Test transform - only normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST dataset with augmentation
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use OneCycleLR scheduler for better convergence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    # Training loop
    best_test_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        
        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0.0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = 100 * correct / total
            avg_test_loss = test_loss / len(test_loader)
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
            
            # Save best model
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best model saved with accuracy: {test_accuracy:.2f}%')
        
        print('-' * 60)

    print(f'Best Test Accuracy: {best_test_accuracy:.2f}%')
    return model

if __name__ == "__main__":
    train_model() 