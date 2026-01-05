# Iranian Customer Churn Prediction Project

This project implements a Multi-Layer Perceptron (MLP) Neural Network to predict customer churn using the **Iranian Churn Dataset**. It explores different model architectures, optimizers, and loss functions to identify the most effective configuration for binary classification.

## 📋 Table of Contents
* [Project Overview](#project-overview)
* [Data Preprocessing](#data-preprocessing)
* [Model Architectures](#model-architectures)
* [Installation & Setup](#installation--setup)
* [Visualizations](#visualizations)

---

## Project Overview
The goal of this project is to predict whether a customer will churn based on 12 features, including call failure, usage frequency, and customer value. The project compares three different Neural Network configurations:
1.  **Model 1:** Adam Optimizer + Mean Squared Error (MSE) Loss.
2.  **Model 2:** SGD Optimizer + Binary Cross-Entropy (BCE) Loss.
3.  **Model 3:** Adam Optimizer + MSE Loss + Extra Hidden Layer (16 neurons → 8 neurons).

## Data Preprocessing
The script performs a rigorous preprocessing pipeline to ensure data quality:

**Cleaning:** Column names are standardized to lowercase and underscores

**Deduplication:** Exact duplicate rows are removed to prevent bias

**Feature Engineering:** One-hot encoding is applied to categorical features (`complains`, `tariff_plan`, `status`), and redundant columns like `age_group` are dropped.

**Outlier Treatment:** Applied **Log Transformation** (`np.log1p`) to highly skewed usage data (e.g., `seconds_of_use`, `frequency_of_sms`) to normalize distributions.

**Scaling:** Used **Min-Max Scaling** to bring all features into a uniform range of [0, 1]

**Splitting:** A stratified 80/20 split is used to maintain the class ratio in both training and testing sets.

## Model Architectures
All models utilize **ReLU** activation for hidden layers and **Sigmoid** for the output layer to provide probability-based predictions.

| Model   | Hidden Layers            | Activation | Activation (Output) | Optimizer | Loss Function                     | Epochs |
|--------|--------------------------|------------|---------------------|-----------|----------------------------------|--------|
| Model 1 | 1 (8 neurons)           | ReLU       | Sigmoid             | Adam      | Mean Squared Error (MSE)          | 500    |
| Model 2 | 1 (8 neurons)           | ReLU       | Sigmoid             | SGD       | Binary Cross-Entropy (BCE)        | 500    |
| Model 3 | 2 (16 → 8 neurons)      | ReLU       | Sigmoid             | Adam      | Mean Squared Error (MSE)          | 500    |
## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
2. **Install requirements:**
    ```bash
    pip install -r requirements.txt
3. **Run Project**:
    ```bash
    python main.py

## Visualizations
The project generates three key plots for each model to evaluate performance:

**Loss Curve:** Tracks Training vs. Validation loss over epochs to detect overfitting.

**Accuracy Graph:** Visualizes how the success rate improves over time

**Confusion Matrix:** Provides a heatmap of True Positives, False Positives, True Negatives, and False Negatives to assess business impact.
