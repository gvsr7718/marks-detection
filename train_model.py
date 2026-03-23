"""
Training Script for AlexNet Digit Recognizer
Trains on MNIST + EMNIST datasets for handwritten digit recognition.

Based on: "AI-Powered Mark Recognition in Assessment and Attainment Calculation"
         by J. Annrose et al. (ICTACT, Jan 2025)

Usage:
    python train_model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import os
import time

from digit_recognizer import AlexNetDigit


def get_datasets():
    """
    Load MNIST and EMNIST Digits datasets.
    Paper uses both for training to improve robustness.
    """
    # Data augmentation for better handwriting generalization
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # MNIST dataset
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # EMNIST Digits dataset (Paper mentions using EMNIST too)
    try:
        emnist_train = torchvision.datasets.EMNIST(
            root='./data', split='digits', train=True, download=True,
            transform=train_transform
        )
        emnist_test = torchvision.datasets.EMNIST(
            root='./data', split='digits', train=False, download=True,
            transform=test_transform
        )
        
        # Combine MNIST + EMNIST
        train_dataset = ConcatDataset([mnist_train, emnist_train])
        test_dataset = ConcatDataset([mnist_test, emnist_test])
        print(f"Combined dataset: {len(train_dataset)} train, {len(test_dataset)} test")
    except Exception as e:
        print(f"EMNIST download failed: {e}. Using MNIST only.")
        train_dataset = mnist_train
        test_dataset = mnist_test
    
    return train_dataset, test_dataset


def train():
    """
    Train the AlexNet model.
    
    Uses Adam optimizer (Paper §3.1) to reduce cross-entropy loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 15  # Paper mentions 15th epoch achieving best accuracy
    
    # Load data
    train_dataset, test_dataset = get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=2)
    
    # Initialize model
    model = AlexNetDigit(num_classes=10).to(device)
    
    # Adam optimizer (Paper §3.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_accuracy = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}], "
                      f"Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch [{epoch}/{num_epochs}] - "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", "alexnet_digits.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f"  ✓ New best model saved! Accuracy: {test_acc:.2f}%")
        
        scheduler.step()
    
    print("-" * 60)
    print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: models/alexnet_digits.pth")


def evaluate(model, test_loader, device):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


if __name__ == "__main__":
    train()
