# Project Overview

This project focuses on training multiple custom deep learning models for performing binary classification on gel electrophoresis images. The project utilizes transfer learning technique via feature extraction and using the PyTorch deep learning framework within the Google Colab environment.

The goal of this project is to investigate and compare the performance of three popular pretrained CNN model architectures: ResNet18, VGG16, and MobileNetV3. A total of 24 models are constructed, each utilizing the architecture of these pretrained models as a backbone. 
The comparison is conducted based on the following factors:

1- Hardware used for training: The models are trained using both CPU and TPU.

2- Classifier head design: Two different designs are explored for the classifier head - a single linear layer and multiple linear layers. 
This comparison aims to evaluate the impact of the classifier head architecture on the models' classification performance.

3- Optimization algorithm: Two widely used optimization algorithms, Adam and SGD, are employed during training. 
Comparing the models trained with these algorithms enables an assessment of their effect on the models' accuracy and overall performance.

By considering these factors, the project aims to provide a comprehensive analysis of the performance of the pretrained models under various configurations.


## There are a total of 24 models developed in this project:

* 3 pretrained model architectures:

  	1- ResNet18
    2- VGG16
    3- MobileNetV3

* 2 classifier head architectures for each pretrained model:

   1- Single linear layer
   2- Multiple linear layers

* 2 optimization algorithms for each classifier head/pretrained model combination:

  1- Adam
  2- SGD

* 2 hardware types for each configuration:

  1- CPU
  2- TPU

Breakdown by model:

* ResNet18
  * 8 models (2 classifier heads x 2 optimizers x 2 hardware)

* VGG16
  * 8 models (2 classifier heads x 2 optimizers x 2 hardware)

* MobileNetV3
  * 8 models (2 classifier heads x 2 optimizers x 2 hardware)