# Thyroid Nodule Classification using VGG19

This project implements a deep learning pipeline using the VGG19 convolutional neural network to classify thyroid nodules in scintigraphy images as **benign** or **malignant**. The model aids in early and accurate diagnosis of thyroid conditions, potentially supporting clinical decision-making.

## Overview

- **Model Architecture:** VGG19 (pretrained on ImageNet, fine-tuned for binary classification)
- **Dataset:** Thyroid scintigraphy images with labeled benign and malignant cases
- **Objective:** Binary classification of thyroid nodules to assist radiological diagnosis
- **Framework:** Python, TensorFlow / Keras

## Features

- Image preprocessing and augmentation for robust training
- Transfer learning with VGG19 for feature extraction
- Fine-tuned classification head for medical imaging task
- Training and validation loss/accuracy tracking
- Evaluation using confusion matrix and classification metrics
