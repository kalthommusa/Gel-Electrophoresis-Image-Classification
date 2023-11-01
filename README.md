# Project Overview

## Goal: 

The goal of this project is to develop and train multiple image classification models using three popular pretrained CNN architectures, ResNet18, VGG16, and MobileNet-V3, with different configurations on a dataset of gel electrophoresis images. The project aims to explore the performance of these models with various settings, including different classifier heads and optimization algorithms. The project utilizes transfer learning technique via feature extraction and using the PyTorch deep learning framework within the Google Colab environment.


## Purpose: 

The purpose of this project is to investigate and compare the performance of different models in classifying grayscale gel images. By training and evaluating 24 models with varying configurations, the project aims to identify the most effective model architecture and configuration for accurate gel image classification.


## Summary of the key points:

* The project generates and builds 24 image classification models by running the train.py script with different command line arguments.

* Each model has a unique combination of:

    * Pretrained model (ResNet18, VGG16, MobileNetV3)

	* Hardware type (CPU, TPU)

	* Classifier head architecture (single linear layer, multiple linear layers)

    * Optimization algorithm (Adam, SGD)

* The models are trained on a dataset of grayscale gel electrophoresis images.

* Training and evaluation results for each model are stored in a results.csv file.

* This allows analysis and comparison of key performance metrics across models, such as:

    * Accuracy

    * Precision

    * Recall

    * ROC AUC

* By systematically varying the model parameters, the project aims to identify the best performing combinations for classifying gel images.


## Breakdown of the 24 models from the configurations perspective: 

1- Hardware Type:

   * 12 models were trained on CPU platform.
   * 12 models were trained on TPU platform.

2- Pretrained Model:

   * 8 models were based on the ResNet18 architecture.
   * 8 models were based on the VGG16 architecture.
   * 8 models were based on the MobileNet-V3 architecture.

3- Classifier Head:

   * 12 models had a single linear layer as the classifier head.
   * 12 models had multiple linear layers as the classifier head.

4- Optimization Algorithm:

   * 12 models used the Adam optimizer algorithm.
   * 12 models used the SGD optimizer algorithm.


## Folders:

* `dataset`: This folder contains the labelled images used to train the models. The images are separated into two subfolders based on their class (gel, not_gel).

* `test_images`: This folder contains images used for inference and evaluating the trained models.


## Files:

The project contains 4 core Python script files that work together to build, train and evaluate the deep learning models for the task of classifying gel electrophoresis images:


* `utils.py`: Contains utility functions used throughout the project. It includes data transformation functions for training, validation, and testing, as well as functions for training and validating the model, saving plots of the training progress, and saving the results to a CSV file.


* `build_dataset.py`: Contains code to programmatically load the dataset, preprocess the images, and split the data into training and validation sets. It defines a custom Dataset class to handle loading and preprocessing the images in an optimized way.


* `prepare_model.py`: Implements functions to initialize popular pre-trained CNN architectures (ResNet18, VGG16, MobileNetV3) from PyTorch, modify them to suit the task (grayscale input, customized classification head), and return the modified model objects.


* `train.py`: Implements the main training loop logic (the training process of the chosen model). It handles command line parameter parsing, model selection, data loading, training, validation, metric tracking, model saving and results logging. 


Together, these files define a complete and modular workflow to efficiently build, tune and evaluate various deep learning models on this image classification benchmark in a structured, comparable manner. 


* `results.csv`: This CSV file stores the results of different model training experiments. It contains information such as the model ID, model architecture, training parameters, and evaluation metric for each experiment. The file is updated with new results each time a model is trained and evaluated.