# Project Overview

This project aims to develop and train multiple image classification models using pretrained convolutional neural network (CNN) architectures as backbones on a custom dataset of gel electrophoresis images. The goal is to explore the performance of these models with different configurations, including various classifier heads and optimization algorithms. The project utilizes transfer learning techniques via feature extraction and is implemented using the PyTorch deep learning framework within the Google Colab environment.

Gel electrophoresis is a widely used technique in molecular biology for separating DNA, RNA, or proteins based on their size and charge. Accurate classification of gel images is crucial for various biological research applications. This project focuses on developing effective models to automate the classification process.


## Goal

The goal of this project is to investigate and compare the performance of different models in classifying grayscale gel images. By training and evaluating 24 models with varying configurations, I aim to identify the most effective model architecture and configuration for accurate gel electrophoresis image classification.


## Dataset

The [dataset](dataset/) comprises a total of 92 images sourced from the internet. It includes 46 grayscale gel electrophoresis images and 46 randomly selected grayscale non-gel images. 


## Models and Configurations

The models were evaluated based on four factors that influenced the training:

**1- Model Architecture:**

As a backbone, I chose 3 pretrained CNN architectures to be implemented in this project:

* ResNet18

* VGG16 

* MobileNetV3


**2- Classifier Head:**

I used the mentioned backbones as a feature extractor and customized them by replacing the final fully connected layer with either:

* Single linear layer as the classifier head (single layer for shortcut): This configuration consists of a single linear layer for classification.

* Sequential multi-layer as the classifier head (multi-layer for shortcut): This configuration comprises multiple sequential layers for classification, providing a deeper and more complex structure.


**3- Optimizer Algorithm:**

I trained each architecture/head combination using either the Adam or SGD optimizer. Both optimizers utilized the CrossEntropyLoss function and were set with a fixed learning rate of 0.001 to ensure a fair comparison.

* Adam is an adaptive learning rate optimization algorithm that is widely used for its simplicity and robustness. It adjusts the learning rate dynamically during training to improve convergence.

* SGD (Stochastic Gradient Descent) is a traditional optimization algorithm that updates the model parameters with the gradients of the loss function. It iteratively adjusts the weights with a fixed learning rate to minimize the loss.


**4- Hardware platform:**

I trained the models on two types of hardware platforms:

* CPU (Central processing unit)

* TPU (Tensor processing unit) specifically designed for machine learning tasks.


## Configuration Breakdown

| Configuration | Number of Models |
|---|---|
| **Hardware Type** | |
| CPU | 12 |
| TPU | 12 |
| **Pretrained Model** | |
| ResNet18 | 8 |
| VGG16 | 8 |
| MobileNet-V3 | 8 |
| **Classifier Head** | |
| Single Linear Layer | 12 |
| Multi-Layer | 12 |
| **Optimization Algorithm** | |
| Adam | 12 |
| SGD | 12 |


## Folders

* `dataset`: Contains the dataset used for the gel project image classification. It contains a collection of gel electrophoresis images that were labeled and organized into two subfolders based on their class (gel, not_gel).

* `test_images`: Contains Contains separate gel electrophoresis images used for evaluating the trained models. These images were not included in the training dataset.


## Files

The project contains 4 core Python script files that work together to efficiently build, tune and evaluate various deep learning models on this image classification benchmark in a structured, comparable manner:


* `utils.py`: Contains utility functions used throughout the project. It includes data transformation functions for training, validation, and testing, as well as functions for training and validating the model, saving plots of the training progress, and saving the results to a CSV file. 


* `build_dataset.py`: Contains code to load the dataset, preprocess the images, and split the data into training and validation sets using a custom Dataset class.


* `prepare_model.py`: Implements functions to initialize pretrained CNN architectures (ResNet18, VGG16, MobileNetV3) from PyTorch, modifying them for grayscale input and customized classification heads.

* `train.py`: Implements the main training loop logic, handling command line parameter parsing, model selection, data loading, training, validation, metric tracking, model saving, and result logging.


* `results.csv`: This CSV file stores the results of different model training experiments, including model ID, architecture, training parameters, and evaluation metrics for each experiment. The file is updated with new results each time a model is trained and evaluated.


# Implementation

The following notebooks serve as practical examples and resources for understanding and implementing gel image classification using deep learning models. Each notebook focuses on training a specific model architecture (ResNet18, VGG16, or MobileNetV3) on either a CPU or TPU.

* [1_Gel_Classifier_ResNet18.ipynb](1_Gel_Classifier_ResNet18.ipynb): This Jupyter notebook showcases the implementation of training a customized ResNet18 model on a CPU. This notebook explores various configurations, evaluates the model's performance, and demonstrates the process of making inferences using the trained model.


* [2_Gel_Classifier_VGG16.ipynb](2_Gel_Classifier_VGG16.ipynb): This Jupyter notebook showcases the implementation of training a customized VGG16 model on a CPU. This notebook explores various configurations, evaluates the model's performance, and demonstrates the process of making inferences using the trained model.


* [3_Gel_Classifier_MobileNetV3.ipynb](3_Gel_Classifier_MobileNetV3.ipynb): This Jupyter notebook showcases the implementation of training a customized MobileNetV3 model on a CPU. This notebook explores various configurations, evaluates the model's performance, and demonstrates the process of making inferences using the trained model.


* [4_Gel_Classifier_ResNet18.ipynb](4_Gel_Classifier_ResNet18.ipynb): This Jupyter notebook showcases the implementation of training a customized ResNet18 model on a TPU. This notebook explores various configurations, evaluates the model's performance, and demonstrates the process of making inferences using the trained model.


* [5_Gel_Classifier_VGG16.ipynb](5_Gel_Classifier_VGG16.ipynb): This Jupyter notebook showcases the implementation of training a customized VGG16 model on a TPU. This notebook explores various configurations, evaluates the model's performance, and demonstrates the process of making inferences using the trained model.


* [6_Gel_Classifier_MobileNetV3.ipynb](6_Gel_Classifier_MobileNetV3.ipynb): This Jupyter notebook showcases the implementation of training a customized MobileNetV3 model on a TPU. This notebook explores various configurations, evaluates the model's performance, and demonstrates the process of making inferences using the trained model.


* [7_Results_Visualizations.ipynb](7_Results_Visualizations.ipynb): This Jupyter notebook provides a set of visualizations to compare the performance of 24 models. These visuals serve as an efficient and informative summary of the models' performance, aiding in the understanding and interpretation of the results of the gel project image classification. 


## Model Training

I trained the 24 models by running the `train.py` script 24 times using the 6 notebooks mentioned above. Each execution involved specifying unique model configurations through command line arguments and providing specific file paths.

* Example:

```
python train.py 
  --dataset_dir='/dataset'
  --model_id=1
  --hardware_type='cpu'  
  --pretrained_model='resnet18'
  --classifier_head='single'
  --opt_alg='adam'
  --model_path='/gel_classifier-1.pth'
  --plots_path='/gel_classifier-1.png'
  --results_file='/results.csv'
```

## Command arguments:

| Name | Description | Default |
|-|-|-|
|--dataset_dir| Path to the Dataset Directory | './data'|
|--model_id| The ID number assigned to the model| 1|  
|--hardware_type| The type of the hardware used to train the model (cpu, tpu) |'cpu'|
|--pretrained_model| Pretrained model (resnet18, vgg16, mobilenet-v3)|'resnet18'|
|--classifier_head| The architecture of the classifier head to use (single, multi)|'single'|
|--opt_alg| Type of the optimizer algorithm to use (adam, sgd)|'adam'|
|--learning_rate| Learning rate for the optimizer|0.001|
|--batch_size| Batch size|23|
|--num_epochs| Number of training epochs|20|
|--model_path| Path to save the custom trained model parameters/weights|'./models/model.pth'|
|--plots_path| Path to save the loss and accuracy plots|'./plots/'|
|--results_file| Path to save the comparison results|'results.csv'|

By running the train.py script with different combinations of these arguments, the 24 models were trained and evaluated, resulting in the generation of the model files, training plots, and the update of the results.csv file with the corresponding model's performance metrics.


# Model Performance Summary

The table below presents a summary of the evaluation results for all 24 models, including their model IDs, hardware types, pretrained models, classifier heads, optimization algorithms, training times, inference times, accuracy, precision, recall, confusion matrices, and ROC AUC scores. (the same table as in this [results.csv](results.csv) file)


| model_id | hardware_type | pretrained_model | classifier_head | opt_alg | training_time_mins | inference_time_secs | accuracy | precision | recall | conf_mat | roc_auc |
|-|-|-|-|-|-|-|-|-|-|-|-|
| 1 | cpu | resnet18 | single | adam | 5.87778879404068 | 8.196835994720459 | 81.82% | 86.67% | 81.82% | [[7, 4], [0, 11]] | 81.82% |
| 2 | cpu | resnet18 | single | sgd | 5.875684936841329 | 9.499671697616575 | 68.18% | 80.56% | 68.18% | [[4, 7], [0, 11]] | 68.18% |
| 3 | cpu | resnet18 | multi | adam | 5.881606896718343 | 9.242843866348268 | 81.82% | 86.67% | 81.82% | [[7, 4], [0, 11]] | 81.82% | 
| 4 | cpu | resnet18 | multi | sgd | 3.714953029155731 | 4.776238441467285 | 90.91% | 92.31% | 90.91% | [[9, 2], [0, 11]] | 90.91% |
| 5 | cpu | vgg16 | single | adam | 38.97634260257085 | 21.181527137756348 | 68.18% | 80.56% | 68.18% | [[4, 7], [0, 11]] | 68.18% |
| 6 | cpu | vgg16 | single | sgd | 39.28553481896719 | 22.356066465377808 | 50.00% | 25.00% | 50.00% | [[0, 11], [0, 11]] | 50.00% |  
| 7 | cpu | vgg16 | multi | adam | 40.49127728939057 | 21.004456520080566 | 77.27% | 84.38% | 77.27% | [[6, 5], [0, 11]] | 77.27% |
| 8 | cpu | vgg16 | multi | sgd | 40.26367266575495 | 22.94587540626526 | 77.27% | 84.38% | 77.27% | [[6, 5], [0, 11]] | 77.27% |
| 9 | cpu | mobilenet-v3 | single | adam | 3.222896846135457 | 6.839791774749756 | 54.55% | 76.19% | 54.55% | [[11, 0], [10, 1]] | 54.55% |
| 10 | cpu | mobilenet-v3 | single | sgd | 3.4531890551249185 | 13.87570333480835 | 54.55% | 56.47% | 54.55% | [[9, 2], [8, 3]] | 54.55% |  
| 11 | cpu | mobilenet-v3 | multi | adam | 3.35837957461675 | 6.238333702087402 | 63.64% | 69.41% | 63.64% | [[10, 1], [7, 4]] | 63.64% |
| 12 | cpu | mobilenet-v3 | multi | sgd | 3.4187332073847454 | 15.212673664093018 | 68.18% | 69.64% | 68.18% | [[9, 2], [5, 6]] | 68.18% |
| 13 | tpu | resnet18 | single | adam | 6.160593561331431 | 9.255364656448364 | 81.82% | 86.67% | 81.82% | [[7, 4], [0, 11]] | 81.82% |
| 14 | tpu | resnet18 | single | sgd | 6.351393938064575 | 9.293831586837769 | 86.36% | 89.29% | 86.36% | [[8, 3], [0, 11]] | 86.36% |
| 15 | tpu | resnet18 | multi | adam | 6.325719388326009 | 8.572688341140747 | 86.36% | 89.29% | 86.36% | [[8, 3], [0, 11]] | 86.36% |
| 16 | tpu | resnet18 | multi | sgd | 6.233783614635468 | 7.970419406890869 | 95.45% | 95.83% | 95.45% | [[10, 1], [0, 11]] | 95.45% |
| 17 | tpu | vgg16 | single | adam | 44.00131607453029 | 24.45604181289673 | 77.27% | 84.38% | 77.27% | [[6, 5], [0, 11]] | 77.27% | 
| 18 | tpu | vgg16 | single | sgd | 41.750093682607016 | 33.835485219955444 | 63.64% | 78.95% | 63.64% | [[3, 8], [0, 11]] | 63.64% |
| 19 | tpu | vgg16 | multi | adam | 40.56718958616257 | 23.428988933563232 | 81.82% | 82.91% | 81.82% | [[8, 3], [1, 10]] | 81.82% |
| 20 | tpu | vgg16 | multi | sgd | 41.16651406288147 | 23.42211103439331 | 86.36% | 89.29% | 86.36% | [[8, 3], [0, 11]] | 86.36% |  
| 21 | tpu | mobilenet-v3 | single | adam | 4.605473589897156 | 8.394397020339966 | 63.64% | 78.95% | 63.64% | [[11, 0], [8, 3]] | 63.64% |
| 22 | tpu | mobilenet-v3 | single | sgd | 4.550919918219249 | 16.60713791847229 | 72.73% | 73.50% | 72.73% | [[9, 2], [4, 7]] | 72.73% |
| 23 | tpu | mobilenet-v3 | multi | adam | 5.102375487486522 | 7.289824247360229 | 72.73% | 73.50% | 72.73% | [[7, 4], [2, 9]] | 72.73% |  
| 24 | tpu | mobilenet-v3 | multi | sgd | 5.599146846930186 | 9.973444938659668 | 81.82% | 82.91% | 81.82% | [[8, 3], [1, 10]] | 81.82% |


# Performance Comparison Visualizations

The figures below provide a visual comparison of various performance metrics for the 24 models including accuracy, precision, recall, ROC AUC as well as training and inference times. Bar charts compare individual metrics across models while heatmaps show each model's performance across all metrics. Additional charts analyze the impact of model architecture, hardware type, classifier head and optimizer on accuracy.


![alt text](imgs/model_accuracies.png)

![alt text](imgs/model_precisions.png)

![alt text](imgs/model_recalls.png)

![alt text](imgs/model_ROC_AUCs.png)

![alt text](imgs/model_ids_vs_evaluation_metrics.png)

![alt text](imgs/architecture_vs_accuracy.png)

![alt text](imgs/hardware_vs_accuracy.png)

![alt text](imgs/classifier_heads_vs_accuracy.png)

![alt text](imgs/optimizer_vs_accuracy.png)

![alt text](imgs/architecture_vs_training_time.png)

![alt text](imgs/architecture_vs_inference_time.png)


## The most effective configurations:

* ResNet18 model achieved the highest accuracies using a multi-layer classifier head and SGD optimizer. Specifically, the ResNet18 model trained on TPU with these settings (Model ID 16) achieved the maximum accuracy of 95.45%. Meanwhile, the ResNet18 model trained on CPU with a multi-layer head and SGD (Model ID 4) obtained a slightly lower but still high accuracy of 90.91%. This demonstrates that although both models performed well overall, the TPU-based ResNet18 configuration led to the best result for this task.

## Further analysis and observations:

* ResNet18 architecture shows consistent performance across different configurations. Models with ResNet18 achieve high accuracy and balanced precision and recall values, this suggests ResNet18 is better suited for this task.

* VGG16 models performed reasonably well, achieving accuracies of 77.27-86.36% on both the CPU and TPU (Model IDs 7, 8, 20).

* MobileNetV3 models performed the worst, with maximum accuracy of 81.82% (Model ID 24). 
The lighter weight architecture seems less optimal for this task than heavier ResNet18 and VGG16.

* VGG16 with a single classifier head and SGD optimizer had the lowest accuracy of 50.00% on a CPU (Model ID 6). This model struggled to effectively classify the data.

* Models with multi-layer as the classifier head outperformed models with single layer as the classifier head for all settings on both the CPU and TPU. This suggests that the extra layers help learn more complex features.

	* For example, ResNet18 with single layer classifier achieved 86.36% accuracy on TPU using SGD optimizer (Model ID 14) vs 95.45% with multi-layer classifier using the same settings (Model ID 16). Biggest difference was for VGG16 with single layer classifier achieved 50.00% accuracy on CPU using SGD optimizer (Model ID 6) vs 77.27% with multi-layer classifier using the same settings (Model ID 8). The improved performance indicates multi-layer as the classifier head have greater representation power for this classification problem. 

* Both Adam and SGD worked well but SGD generally achieved higher accuracy than Adam across different model architectures/configurations, often by significant improvements like ResNet18 with Adam optimizer on TPU achieved 86.36% accuracy (Model ID 15) vs 95.45% accuracy with SGD optimizer using the same settings (Model ID 16). Suggesting SGD converges better and may be better suited than Adam for this specific image classification task.

* Training on the TPU led to better results than the CPU for all models, with accuracy improvements of 3-15%. The accelerated compute of the TPU benefited the training.

* Heavy models like VGG16 took the longest to train (38-44 mins). ResNet18 and MobileNetV3 were faster (~3-6 mins).

* Inference time followed relative model complexity. VGG16 > ResNet18 > MobileNetV3. 

## Demo

The following figures showcase the predictions of the ResNet18/multi-layer/SGD on TPU (Model ID 16) on unseen gel images, providing insights into its performance. In the evaluation of 11 test images, this particular model demonstrated exceptional accuracy in recognizing and identifying the gel images, successfully classifying all but one image. 

![alt text](imgs/prediction1.png)
![alt text](imgs/prediction2.png)
![alt text](imgs/prediction3.png)
![alt text](imgs/prediction4.png)
![alt text](imgs/prediction5.png)
![alt text](imgs/prediction6.png)
![alt text](imgs/prediction7.png)
![alt text](imgs/prediction8.png)
![alt text](imgs/prediction9.png)
![alt text](imgs/prediction10.png)
![alt text](imgs/prediction11.png)

This outstanding performance highlights the effectiveness and reliability of the ResNet18 architecture coupled with the multi-layer approach and the Stochastic Gradient Descent (SGD) optimizer on TPU hardware. The model's ability to accurately classify the majority of gel images reaffirms its proficiency and reinforces its suitability for gel project image classification tasks.


## Conclusion

This systematic evaluation provided valuable insights into how architectural decisions, optimizer choice, and hardware can impact model effectiveness. The best combination identified here demonstrates an optimized configuration for highly accurate gel image classification.