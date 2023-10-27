import torch.nn as nn
import torchvision.models as models

def resnet18(num_classes, classifier_head='single'):
    """
    Create a modified ResNet18 model pretrained on ImageNet with a customized model head.
    
    Args:
        num_classes (int): Number of classes for the classification task.
        classifier_head (str): Type of model head architecture (single, multi). 

    Returns:
        torch.nn.Module: Modified ResNet18 model.
    """
    
    # Load ResNet18 model pretrained on ImageNet
    model = models.resnet18(pretrained=True)

    # Freeze early layers of ResNet18 model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the first input conv layer to accept grayscale images(1 channel)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Get classifier input features
    num_ftrs = model.fc.in_features

    # Define the architecture of the custom model head based on the classifier_head flag
    if classifier_head == 'single':
        # Use single layer classifier
        model.fc = nn.Linear(num_ftrs, num_classes) # Replace the last fully connected layer with a new one

    elif classifier_head == 'multi':
        # Use multi-layer classifier
        classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model.fc = classifier # Replace the last fully connected layer with a new one

    else:
        raise ValueError('Invalid classifier architecture')

    return model


def vgg16(num_classes, classifier_head='single'):
    """
    Create a modified VGG16 model pretrained on ImageNet with a customized model head.

    Args:
        num_classes (int): Number of classes for the classification task.
        classifier_head (str): Type of model head architecture (single, multi). 

    Returns:
        torch.nn.Module: Modified VGG16 model with a customized model head.
    """

    # Load pretrained VGG16
    model = models.vgg16(pretrained=True)

    # Freeze early layers of VGG16 model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the first input conv layer to accept grayscale images(1 channel)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

    # Get classifier input features
    num_ftrs = model.classifier[6].in_features

    # Define the architecture of the custom model head based on the classifier_head flag
    if classifier_head == 'single':
        # Use single layer classifier
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) # Replace the last fully connected layer with a new one

    elif classifier_head == 'multi':
        # Use multi-layer classifier
        classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model.classifier[6] = classifier # Replace the last fully connected layer with a new one

    else:
        raise ValueError('Invalid classifier architecture')

    return model


def mobilenet_v3(num_classes, classifier_head='single'):
    """
    Create a modified MobileNetV3 model pretrained on ImageNet with a customized model head.

    Args:
        num_classes (int): Number of classes for the classification task.
        classifier_head (str): Type of model head architecture (single, multi). 

    Returns:
        torch.nn.Module: Modified MobileNetV3 model with a customized model head.
    """
    # Load pretrained MobileNetV3
    model = models.mobilenet_v3_large(pretrained=True)

    # Freeze early layers of MobileNetV3 model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the first input conv layer to accept grayscale images(1 channel)
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

    # Get classifier input features
    num_ftrs = model.classifier[3].in_features

    # Define the architecture of the custom model head based on the classifier_head flag
    if classifier_head == 'single':
        # Use single layer classifier
        model.classifier[3] = nn.Linear(num_ftrs, num_classes) # Replace the last fully connected layer with a new one

    elif classifier_head == 'multi':
        # Use multi-layer classifier
        classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model.classifier[3] = classifier # Replace the last fully connected layer with a new one

    else:
        raise ValueError('Invalid classifier architecture')

    return model
    