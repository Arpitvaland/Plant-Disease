# Plant-Disease

## Abstract

This report outlines a deep learning approach for plant disease classification using DenseNet121. Employing transfer learning, the pre-trained architecture is fine-tuned on a dataset of plant images with diseases. Data preprocessing involves resizing, normalization, and augmentation. Training utilizes an Adam optimizer (lr: 0.002) for 50 epochs with a batch size of 64. Performance is evaluated using accuracy and loss metrics, enhancing the model via learning rate annealing and checkpoints. The resulting model achieves a validation accuracy of around 89.41%, facilitating effective plant disease identification.

## I. Introduction

Plant diseases impact global food security, necessitating rapid and accurate detection for minimal crop losses. This project leverages deep learning, specifically DenseNet121, to automate plant disease classification. The model is trained on an extensive dataset of plant images to differentiate between diseases, aiding in real-world identification and management.

## II. Literature Review

### A. Plant Disease Detection and Classification

Traditional methods of plant disease diagnosis are time-consuming and labor-intensive. Machine learning, particularly Convolutional Neural Networks (CNNs), addresses this by automating disease identification based on digital images.

### B. Transfer Learning and Pretrained Models

Transfer learning enhances efficiency by leveraging knowledge from one task to improve another. Pretrained models, such as VGG, ResNet, Inception, and DenseNet, fine-tune on domain-specific datasets, contributing to accurate disease classification.

### C. Data Augmentation and Preprocessing

Data augmentation addresses limited training data by expanding datasets through transformations. Proper preprocessing ensures model robustness, including resizing images, normalizing pixel values, and removing noise.

### D. Evaluation Metrics

Evaluation metrics like accuracy, precision, recall, and F1-score provide insights into model performance. The confusion matrix offers a detailed view of classification outcomes.

### E. Related Studies

Successful applications of deep learning in plant disease classification highlight the technology's potential. CNNs distinguish between various plant diseases with high accuracy, and smartphone applications enable real-time disease diagnosis.

## III. Methodology

### A. Data Preprocessing

Data preprocessing involves augmentation, resizing images to 224x224 pixels, and normalization (scaling pixel values to [0, 1]).

### B. Model Selection and Architecture

DenseNet121 is selected for its dense connectivity pattern, efficient parameter use, and effectiveness in capturing intricate patterns.

### C. Training Strategy

The dataset is split into training, validation, and testing sets. Training uses the Adam optimizer, categorical cross-entropy loss, and a learning rate scheduler.

### D. Evaluation Metrics

Various metrics, including accuracy, precision, recall, F1-score, and the confusion matrix, assess model performance.

## IV. Experiments

### A. Experiment Design

Experiments utilize a machine with an AMD Ryzen5 3550H processor, 16GB RAM, and an NVIDIA GeForce GTX graphics card. TensorFlow and Keras frameworks implement the DenseNet-121 architecture.

### B. Dataset Preparation

Dataset preparation involves understanding class distribution, dividing into sets, and employing data augmentation techniques.

### C. Evaluation Metrics

Metrics include the classification report and accuracy for a comprehensive understanding of model performance.

### D. Results and Analyses

Experiments involve hyperparameter selection, model evaluation, and a Streamlit web application showcasing the model's practical application.

