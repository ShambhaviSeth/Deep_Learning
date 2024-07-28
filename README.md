# Deep Learning Projects

Welcome to the Dee Learning repository! This repository contains various deep learning projects and implementations aimed at enhancing our understanding and skills in the field of deep learning. Each project is designed to tackle specific problems and datasets, providing a hands-on approach to learning and mastering deep learning techniques.

## Projects

  **1. Analyzing Movie Reviews using Transformers**


This project involves training a sentiment analysis model using the BERT (Bidirectional Encoder Representations from Transformers)   model. We will parse movie reviews and classify their sentiment (positive or negative). The Huggingface transformers library is used to load a pre-trained BERT model for text embeddings, which is then combined with an RNN model for sentiment classification.

**Key Features:**
  - Utilizes BERT for text embeddings.
  - Implements an RNN model for sentiment classification.
  - Achieves high accuracy in sentiment analysis tasks.

**2. Backpropagation from Scratch**

This project demonstrates how to train a neural network from scratch using only Numpy. It involves implementing backpropagation manually and testing the network on the MNIST dataset, which consists of images of handwritten digits. Additionally, it includes implementing backpropagation for more complicated operations directly, rather than using autodiff.

**Key Features:**

  - No use of deep learning libraries (e.g., TensorFlow, PyTorch).
  - Manual implementation of forward and backward propagation.
  - Insight into the inner workings of neural networks.

**3. Transformers in Computer Vision**

This project explores the application of transformer architectures to computer vision tasks. Specifically, it involves developing a Vision Transformer (ViT) model for processing image data. The ViT model extracts patches from images, treats them as tokens, and passes them through a sequence of transformer blocks before using dense classification layers.

**Key Features:**

  - Utilizes transformer architectures for image processing.
  - Implements the Vision Transformer (ViT) model.
  - Demonstrates the performance of transformers in computer vision tasks.

**4. Improving the FashionMNIST Classifier**

This project aims to improve the classification accuracy of the FashionMNIST dataset by using a dense neural network with three hidden layers (256, 128, and 64 neurons) and ReLU activations. The project includes training and testing the model, displaying train- and test-loss curves, and reporting test accuracies. Additionally, it visualizes the predicted class probabilities for selected test samples.

**Key Features:**

  - Utilizes a neural network with three hidden layers.
  - Implements dropout and batch normalization for better generalization.
  - Achieves significant accuracy improvements over baseline models.

