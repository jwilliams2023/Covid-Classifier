
# COVID Classifier Project

## Overview

This project aims to develop a machine learning model to predict the mortality risk of COVID-19 patients based on their health and demographic data. Accurate predictions can assist healthcare providers in resource allocation and early intervention for high-risk patients.

The script leverages **PyTorch**, a popular deep learning library, to build and train a neural network model. The model predicts whether a COVID-19 patient is at high risk of mortality based on various health indicators.

## Dataset

I utilized a dataset provided by the Mexican government, available on [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data). The dataset contains anonymized information on over one million patients, including:

- **Demographics**: Age, sex.
- **Medical History**: Presence of comorbidities like diabetes, hypertension, obesity, etc.
- **COVID-19 Related Data**: Test results, symptoms, and outcomes.

## How PyTorch and Neural Networks Help

Using **PyTorch**, this project builds a neural network model that can learn complex patterns in the data to predict mortality risk. Neural networks, especially deep learning models, are capable of handling large datasets and capturing nonlinear relationships between features.

## Environment Setup

To ensure reproducibility and manage dependencies, I set up the project environment using Anaconda. Given that I have an NVIDIA RTX 4080 graphics card—which is beneficial for both gaming and high-performance computations—I configured the environment to leverage CUDA for GPU acceleration during model training.

## Data Preprocessing

Data preprocessing was essential to prepare the dataset for modeling. The following steps were undertaken:

1. **Data Cleaning**:
   - Dropped columns with high missing rates or irrelevant information.
   - Removed rows with missing values encoded as specific codes.

2. **Feature Encoding**:
   - Standardized binary categorical variables to a 0 (No) and 1 (Yes) format.
   - Converted the `DATE_DIED` column to a binary target variable representing mortality.

3. **Feature Scaling**:
   - Normalized the `AGE` column to a range between 0 and 1.

4. **Feature Selection and Reordering**:
   - Selected relevant features and reordered columns for clarity.

The cleaned dataset was saved for efficient loading in the model.

## Model Architecture

The model, defined in the `CovidClassifier` class, consists of:

- An **Input Layer** that transforms input features to a 32-dimensional space.
- **Hidden Layers** with ReLU activation functions to capture nonlinear relationships in the data.
- An **Output Layer** that provides the final prediction.

## Model Training

The training script `train.py` configures the model with specific hyperparameters:

- **Epochs**: 20
- **Learning Rate**: 0.0003
- **Batch Size**: 256
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Mean Squared Error (MSE)

The script also includes the use of GPU acceleration through CUDA, leveraging the NVIDIA RTX 4080 card for faster computations.

## Dataset Class (`data/CovidDataset.py`)

The `CovidDataset` class was created to efficiently load and prepare batches of data for training. This class handles:

- Reading the cleaned dataset.
- Separating features and labels.
- Providing data in a format compatible with PyTorch for batch processing.

## Conclusion

This project establishes a robust pipeline for predicting COVID-19 mortality risk, including data preprocessing, model building, and training with GPU acceleration. Future steps include:

1. **Model Evaluation**: Testing on validation and test sets to assess performance.
2. **Hyperparameter Tuning**: Refining model parameters for improved accuracy.
3. **Documentation and Visualization**: Updating the README with final results and creating visualization tools for model interpretation.

