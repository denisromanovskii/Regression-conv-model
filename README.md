<p align="center">
      <img src="https://github.com/denisromanovskii/Regression-conv-model/blob/main/conv_back.png" width="726">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Language-Python_3.12-blue" alt="Python Version">
   <img src="https://img.shields.io/badge/Library-PyTorch_2.6.0-orange" alt="PyTorch Version">
   <img src="https://img.shields.io/badge/GUI-PyQt6-red" alt="PyQt Version">
   <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

## About

This project is a convolutional regression neural network designed to find the center of a square in images. The model utilizes convolutional layers to extract features and then regresses the coordinates of the square’s center. The main goal is to accurately determine the center’s coordinates, which can be useful in various applications such as image processing and computer vision.

## Demo

<p align="center">
      <img src="" alt="demo gif" height="300px">
</p>

## Documentation

### Included Files:

- **-** **`datasetCreator.py`** - This script generates or collects the dataset used for training the model. It includes steps like image generation, annotation, or collection from predefined sources.
- **-** **`datasetPreparation.py`** - This file processes the dataset by performing pre-processing tasks such as resizing, normalization, and splitting the data into training, validation, and test sets.
- **-** **`main.py`** - This script creates the graphical user interface (GUI) for interacting with the model, allowing users to input data, trigger model predictions, and visualize results.
- **-** **`model.py`** - This file defines the convolutional regression neural network architecture, specifying the layers, activation functions, loss functions, and the overall configuration of the model.
- **-** **`testModel.py`** - This script evaluates the performance of the trained model by testing it on the test dataset and calculating relevant metrics like accuracy, mean squared error, or other performance measures.
- **-** **`trainModel.py`** - This module is responsible for training the model. It loads the prepared dataset, configures the model, and runs the training process, including optimization, backpropagation, and model saving.

## Developers

- [Denis Romanovskii](https://github.com/denisromanovskii)

## License
Project Regression-conv-model is destributed under the MIT license.
