# COVID-19 Image Classification Project

## Overview
This project aims to classify images related to COVID-19 using a Convolutional Neural Network (CNN). The model is trained on a dataset of grayscale images categorized into several classes related to COVID-19. We use TensorFlow and Keras to design and train our model, employing data augmentation techniques to improve generalization.

## Dataset
The dataset is organized into a training set within the `Covid19-dataset/train` directory. Images are in grayscale and categorized into multiple classes. For simplicity, we apply data augmentation such as zooming, rotation, and width/height shifting to the training images. Validation data is processed without augmentation.

## Model Architecture
The model is a sequential CNN that includes the following layers:
- Input layer with shape `(256, 256, 1)` for grayscale images.
- Two convolutional layers with ReLU activation, followed by max-pooling layers.
- Dropout layers following each max-pooling layer to reduce overfitting.
- A flattening layer to convert the 2D features to a 1D vector.
- A dense output layer with a softmax activation function to classify 3 different categories.

The model uses the Adam optimizer, categorical crossentropy as the loss function, and tracks categorical accuracy and AUC as metrics.

## Training
Training is conducted with early stopping based on the validation AUC to prevent overfitting. The training process is executed for a predefined number of epochs, with the option to stop early if the validation AUC does not improve for a given number of epochs.

## Evaluation
After training, the model's performance can be evaluated using classification reports and confusion matrices to understand its accuracy, precision, recall, and F1 score across different classes.

## Requirements
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Usage
To run the training script, navigate to the project directory and execute the Python script:

```sh
python covid19detection.py
```

## Contributing
Contributions to the project are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.
