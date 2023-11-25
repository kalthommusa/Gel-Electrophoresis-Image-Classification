# Import the necessary libraries
import utils
import build_dataset
import prepare_model
import os
import csv
import time
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to the Dataset Directory')
    parser.add_argument('--model_id', type=int, default=1, help='The ID number assigned to the model')
    parser.add_argument('--hardware_type', type=str, default='cpu', help='The type of the hardware used to train the model (cpu, gpu)')
    parser.add_argument('--pretrained_model', type=str, default='resnet18', help='Pretrained model (resnet18, vgg16, mobilenet-v3)')
    parser.add_argument('--classifier_head', type=str, default='single', help='The architecture of the classifier head to use (single, multi)')
    parser.add_argument('--opt_alg', type=str, default='adam', help='Type of the optimizer algorithm to use (adam, sgd)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=23, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default='./models/model.pth', help='Path to save the custom trained model parameters/weights')
    parser.add_argument('--plots_path', type=str, default='./plots/', help='Path to save the loss and accuracy plots')
    parser.add_argument('--results_file', type=str, default='results.csv', help='Path to save the comparison results')

    args = parser.parse_args()
    return args

def train_model(args):
    # Get the number of classes
    dataset = build_dataset.GelElectrophoresisDataset(args.dataset_dir)
    num_classes = dataset.num_classes

    # Create training and validation datasets 
    train_dataset, val_dataset = build_dataset.create_train_val_datasets(args.dataset_dir, utils.train_transform, utils.val_transform)

    # Create dataloader iterators for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Choose the pretrained model
    if args.pretrained_model == 'resnet18':
        model = prepare_model.resnet18(num_classes, args.classifier_head)
        print("\n[INFO] The full architecture of ResNet18 model with the customized head:")
        print(model)
    elif args.pretrained_model == 'vgg16':
        model = prepare_model.vgg16(num_classes, args.classifier_head)
        print("\n[INFO] The full architecture of VGG16 model with the customized head:")
        print(model)
    elif args.pretrained_model == 'mobilenet-v3':
        model = prepare_model.mobilenet_v3(num_classes, args.classifier_head)
        print("\n[INFO] The full architecture of MobileNet-V3 model with the customized head:")
        print(model) 
    else:
        raise ValueError("Invalid pretrained model")

    # Choose the device to train on 
    if args.hardware_type == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise ValueError("GPU is not available. Please select 'cpu' as the hardware type.")
    elif args.hardware_type == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError("Invalid hardware type. Please select either 'cpu' or 'gpu'.")
        
    # Move the model to the device
    model = model.to(device)

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer
    if args.opt_alg == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_alg == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=0.9)

    # Initialize lists for training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training Loop
    print("\n[INFO] Start training the model...")
    # Start timer
    train_start = time.time()
    # Loop over epochs
    for epoch in range(args.num_epochs):
        # Train the model on the training data
        train_loss, train_acc = utils.train(model, train_dataloader, criterion, optimizer, device)
        # Validate the model on the validation data
        val_loss, val_acc = utils.validate(model, val_dataloader, criterion, device)

        # Append metrics (the loss and accuracy values) to lists
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print the metrics in each epoch (the loss and accuracy values)
        print(f'Epoch: {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Stop timer after finishing training
    train_end = time.time()
    # Calculate training time
    train_time_secs = train_end - train_start
    # Convert seconds to minutes
    train_time_mins = train_time_secs / 60

    print("[INFO] Finished training the model!")

    # Save the trained model parameter weights
    torch.save(model.state_dict(), args.model_path)
    print(f"[INFO] Model saved at {args.model_path}")

    # Save the loss and accuracy plots
    utils.save_plots(train_losses, val_losses, train_accuracies, val_accuracies, args.plots_path)
    print(f"[INFO] Training and validation loss/accuracy plots saved in {args.plots_path}")

    # Create a dictionary with the training results
    results = {
        'model_id': args.model_id,
        'hardware_type': args.hardware_type,  # Placeholder value for hardware type
        'pretrained_model': args.pretrained_model,
        'classifier_head': args.classifier_head,
        'opt_alg': args.opt_alg,
        'training_time_mins': train_time_mins,
        'inference_time_secs': 0,  # Placeholder value for inference time
        'accuracy': 0,  # Placeholder value for accuracy
        'precision': 0,  # Placeholder value for precision
        'recall': 0,  # Placeholder value for recall
        'conf_mat': None,  # Placeholder value for confusion matrix
        'roc_auc': 0  # Placeholder value for AUC score
    }

    # Save the results to a CSV file
    results_df = pd.DataFrame(results, index=[0])
    utils.save_results(results_df, args.results_file)
    print(f"[INFO] Training time saved in {args.results_file}")
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Train the model
    train_model(args)