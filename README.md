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

This allows analysis and comparison of key performance metrics across models, such as:

    * Accuracy

    * Precision

    * Recall

    * ROC AUC

* By systematically varying the model parameters, the project aims to identify the best performing combinations for classifying gel images.


## Breakdown of the 24 models based on configurations perspective: 

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


## The project involves multiple Python scripts for building the dataset, preparing the model architecture, training the model, and saving the results. The project utilizes popular deep learning frameworks like PyTorch and torchvision to implement the functionality.