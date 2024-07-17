# CV 4022 - Automatic License Plate Recognition Project

This project is part of the Fundamentals of Computer Vision course at KNTU, Spring 2024. The goal is to develop a Convolutional Neural Network (CNN)-based system for detecting and extracting data from Iranian license plates.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Preparation](#dataset-preparation)
3. [Data Augmentation](#data-augmentation)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Dependencies](#dependencies)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

## Introduction

Automatic License Plate Recognition (ALPR) is essential for various applications such as traffic management, law enforcement, electronic toll collection, and security. In this project, we develop a CNN-based system to detect and extract data from Iranian license plates.

## Dataset Preparation

1. **Labeling the Dataset**: The dataset was labeled using Label Studio with the "Semantic Segmentation with Polygons" interface to annotate the four corners of the license plates.

2. **Accessing the Dataset**: The dataset used in this project is not publicly available yet. For access, please contact the course's teacher, B. Nasihatkon, via the [course website](https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2024/) or [personal page](https://wp.kntu.ac.ir/nasihatkon/).

3. **Loading the Dataset**: The images and labels are loaded from the specified directory and preprocessed to be used in training the model.

## Data Augmentation

Data augmentation is applied to increase the diversity of the training set and help reduce overfitting. The augmentation techniques used include:
- Slight shifts
- Blurs
- Noise injection
- Crops
- Rotations
- Contrast adjustments

## Model Architecture

### Regression Model

The regression model is designed to take color images of cars and output the coordinates of the four corners of the license plate. The architecture includes:
- **Convolutional Layers**: Several convolutional layers with ReLU activation to extract features from the input images.
- **Max Pooling Layers**: Applied after some convolutional layers to reduce the spatial dimensions and retain important features.
- **Fully Connected Layers**: Dense layers to process the extracted features and make predictions. Dropout layers are included for regularization.
- **Output Layer**: A final dense layer outputs 8 values, representing the normalized coordinates of the four corners of the license plate.

### Classification Model

The classification model reads the extracted license plate image and identifies the characters. The unique architecture includes:
- **Convolutional Layers**: Multiple convolutional layers with ReLU activation to extract detailed features from the license plate images.
- **Max Pooling Layers**: Applied after convolutional layers to reduce the dimensions while keeping essential features.
- **Reshape Layer**: Reshapes the output to obtain a separate feature vector for each character in the license plate.
- **Fully Connected Layers for Each Character**: Each character's feature vector is processed through individual dense layers.
- **Output Layers**: Each dense layer is connected to an output layer with softmax activation to classify the characters. The third character is treated as a Persian letter, and the rest as digits.

## Training the Model

The models are trained using the following steps:
1. **Splitting the Data**: The dataset is split into training, validation, and test sets.
2. **Compiling the Model**: The models are compiled with the Adam optimizer and mean squared error loss for the regression model, and categorical cross-entropy for the classification model.
3. **Training**: The models are trained with early stopping and model checkpoint callbacks to prevent overfitting and save the best models.

## Evaluation

The models are evaluated on the test set using Mean Absolute Error (MAE) for the regression model and accuracy for the classification model. Visualization functions are used to display the predictions and extracted license plates.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib
- WandB
- Hugging Face Hub

Install the dependencies using:
```bash
pip install tensorflow numpy opencv-python matplotlib wandb huggingface_hub
```

## Acknowledgments

This project is directed by [Mahdi Lotfi](https://github.com/mahdilotfi167) and guided by B. Nasihatkon. Special thanks to the contributors Alireza Honardoost, [Morteza Hajiabadi](https://github.com/m-hajiabadi), and [Kasra Davoodi](https://github.com/the-coding-machine) for their efforts. The course was taught by B. Nasihatkon ([course website](https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2024/)).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Hugging Face Model

The trained regression model is available on Hugging Face: [KNTU-VC-4022-License-Plate-Recognition](https://huggingface.co/AlirezaF138/KNTU-VC-4022-License-Plate-Recognition)
