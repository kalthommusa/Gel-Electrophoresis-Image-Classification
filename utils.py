# Import the necessary libraries
import os
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Reusable data transform functions 
# Transforms for training phase with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Transforms for validation phase without data augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Transforms for test without data augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Reusable model training function
def train(model, dataloader, criterion, optimizer, device):
    """
    Train the model on the training data.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader containing the training data.
        criterion: The loss function.
        optimizer: The optimizer for updating the model's parameters.
        device (torch.device): The device (GPU/CPU) to be used for training.

    Returns:
        tuple: A tuple containing the average loss and accuracy over the epoch.
    """

    # Set the model to train mode
    model.train()

    running_loss = 0.0  # Initialize the running loss
    correct = 0  # Initialize the number of correctly classified samples
    total = 0  # Initialize the total number of samples

    # Iterate over the training data
    for inputs, labels in dataloader:
        # Move the inputs and labels to device (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()
        # Update the running loss
        running_loss += loss.item()
        # Calculate the number of correctly classified samples
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Calculate the average loss and accuracy over the training data
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy

# Reusable model validation function
def validate(model, dataloader, criterion, device):
    """
    Validate the model on the validation data.

    Args:
        model (nn.Module): The model to be validated.
        dataloader (DataLoader): The DataLoader containing the validation data.
        criterion: The loss function.
        device (torch.device): The device (GPU/CPU) to be used for validation.

    Returns:
        tuple: A tuple containing the average loss and accuracy over the validation data.
    """

    # Set the model to evaluation mode
    model.eval()

    running_loss = 0.0  # Initialize the running loss
    correct = 0  # Initialize the number of correctly classified samples
    total = 0  # Initialize the total number of samples

    # Iterate over the validation data
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Update the running loss
            running_loss += loss.item()
            # Calculate the number of correctly classified samples
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate the average loss and accuracy over the validation data
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


def save_plots(train_losses, val_losses, train_accuracies, val_accuracies, plots_path):
    # Save the training/validation loss and accuracy plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(plots_path)


def save_results(results, results_file):
    # Check if the results file already exists
    if os.path.isfile(results_file):
        # Load the existing results
        existing_results = pd.read_csv(results_file)
        # Append the new results as a new row
        #updated_results = existing_results.append(results, ignore_index=True)
        updated_results = pd.concat([existing_results, results], ignore_index=True)
    else:
        # Create a new DataFrame with the results
        updated_results = pd.DataFrame(results, index=[0])

    # Save the results to the CSV file
    updated_results.to_csv(results_file, index=False)
