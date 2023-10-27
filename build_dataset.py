# Import the necessary libraries
import utils
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset 
from torchvision.datasets import ImageFolder

class GelElectrophoresisDataset(Dataset):
    """
    A custom dataset class for Gel Electrophoresis dataset.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            root (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = ImageFolder(root_dir)
        self.samples = self.image_folder.samples
        self.num_classes = len(self.image_folder.classes)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.samples)


    def __getitem__(self, index):
        """
        Get a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        image_path, label = self.samples[index]
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)# Load grayscale image
        gray_image = Image.fromarray(gray_image)
        if self.transform:
            image = self.transform(gray_image)# Apply transformations if provided

        return image, label

def create_train_val_datasets(root_dir, train_transform, val_transform):
    """
    Create training and validation datasets for Gel Electrophoresis.

    Args:
        root_dir (str): Root directory of the dataset.
        train_transform (transform): Data augmentation transformations to be applied to the training data.
        val_transform (transform): transformations to be applied to the validation data.

    Returns:
        tuple: A tuple containing the training dataset and validation dataset.
    """

    # Create training dataset
    train_dataset = GelElectrophoresisDataset(root_dir, transform=train_transform)

    # Create validation dataset
    val_dataset = GelElectrophoresisDataset(root_dir, transform=val_transform)

    # Calculate the size of the training dataset
    train_size = int(0.8 * len(train_dataset))

    # Calculate the size of the validation dataset
    val_size = len(train_dataset) - train_size

    # Randomly split dataset into train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Return the train and val datasets
    return train_dataset, val_dataset
    