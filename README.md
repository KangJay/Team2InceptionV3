# Training a InceptionV3 Image Classification Model

This Microsoft Azure ML sample will take you through the steps of training an image classification model based on the [InceptionV3 Architecture](https://arxiv.org/abs/1512.00567) on the Azure ML platform.

  

## What is InceptionV3?
![Alt text](https://images.deepai.org/glossary-terms/a36d6639cf694722b5cc814528ec0ef6/inception.png)
 Source: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
The primary focus of this architecture is to utilize fewer computational resources in order to develop models quicker. InceptionV3 is a convolutional neural network based off the [InceptionV1 Architecture](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf), InceptionV3 improves upon V1's approach by...
1. Factorizing convolutions into multiple smaller ones.
2. Replacing convolutions with smaller ones. E.g. a 5x5 filter is replaced with two 3x3. A 5x5 filter would have 25 parameters whilst the two 3x3 filters would have 18 parameters total (2 x 3 x 3 = 18). 
3. Asymmetric convolutions. E.g. a 3x3 convolution is replaced by a 1x3 and 3x1 convolutions.
4. Auxiliary classifiers that propagate loss incurred between the convolutional layers. 
5. Grid size reduction


## Getting Started

**Some instructions about how to copy the model, load data, train, and validate on Jupyter Notebooks. WIP**

  

### Prerequisites

> pip install azureml-sdk

  

Read more detailed instructions on [how to set up your environment](https://github.com/Azure/MachineLearningNotebooks/blob/master/NBSETUP.md) using Azure Notebook service, your own Jupyter notebook server, or Docker.