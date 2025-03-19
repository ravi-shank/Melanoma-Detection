## Introduction
Cancer encompasses over 200 distinct types, with melanoma being the deadliest form of skin cancer. The diagnostic process for melanoma typically begins with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection is crucial, as it greatly improves the chances of successful treatment.

The initial step in diagnosing melanoma involves visually examining the affected skin. Dermatologists capture dermoscopic images of skin lesions using high-speed cameras, achieving diagnostic accuracies between 65% and 80% without additional technical support. When combined with further visual assessment by oncologists and dermoscopic image analysis, the overall predictive accuracy increases to 75%–84%.

This project aims to develop an automated classification system that utilizes image processing techniques to classify skin cancer based on images of skin lesions.

## Problem statement

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
The dataset consists of 2,357 images representing both malignant and benign oncological conditions, obtained from the International Skin Imaging Collaboration (ISIC). These images are categorized according to ISIC's classification, ensuring that each subset contains an equal number of samples.

## Design
The final CNN architecture is structured as follows:

Data Augmentation: The augmentation_data variable applies various augmentation techniques to the training dataset. This enhances data diversity by introducing random transformations such as rotation, scaling, and flipping, ultimately improving the model's ability to generalize.

Normalization: A Rescaling(1./255) layer normalizes pixel values, scaling them to a range between 0 and 1. This stabilization aids in faster convergence and a more efficient training process.

Convolutional Layers: The model includes three Conv2D layers, each followed by a ReLU activation function. The padding='same' parameter ensures that the feature maps maintain their original spatial dimensions. The number of filters in each layer (16, 32, 64) determines the depth of extracted features.

Pooling Layers: Each convolutional layer is followed by a MaxPooling2D layer, which reduces the spatial dimensions of feature maps while preserving essential information. This helps control overfitting and decreases computational complexity.

Dropout Layer: A Dropout layer with a rate of 0.2 is included after the last max-pooling layer. This technique randomly drops a fraction of neurons during training to prevent overfitting.

Flatten Layer: The Flatten layer converts 2D feature maps into a 1D vector, preparing the data for the fully connected layers.

Fully Connected Layers: Two dense (Dense) layers are added with ReLU activation functions. The first dense layer contains 128 neurons, while the second outputs class probabilities.

Output Layer: The number of neurons in the output layer corresponds to the number of classes (target_labels). The activation function is omitted here, as the loss function handles the logits during training.

Model Compilation: The model is compiled using the Adam optimizer and the Sparse Categorical Crossentropy loss function, which is suitable for multi-class classification. Accuracy is used as the evaluation metric.

Training Process: The model is trained using the fit method for 50 epochs. EarlyStopping (patience = 5) prevents overfitting by stopping training if validation accuracy stagnates, while ModelCheckpoint saves the model with the best validation accuracy. These callbacks help optimize the final model performance.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- [Python](https://www.python.org/) - version 3.11.4
- [Matplotlib](https://matplotlib.org/) - version 3.7.1
- [Numpy](https://numpy.org/) - version 2.0.32-
- [Pandas](https://pandas.pydata.org/) - version 2.2.2
- [Seaborn](https://seaborn.pydata.org/) - version 0.13.2
- [Tensorflow](https://www.tensorflow.org/) - version 2.18.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- This project was inspired by Upgrad


## Contact
Created by [@ravi-shank] - feel free to contact me!
