import numpy as np
import pandas as pd
import json
import os

# HELPER FUNCTIONS
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def leaky_relu(x, alpha=0.01): 
    return np.where(x > 0, x, alpha * x)

def derivative_leaky_relu(Z, alpha=0.01):
    dZ = np.ones_like(Z)
    dZ[Z <= 0] = alpha
    return dZ

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#
# PREPROCESSING UTILITIES
# 

def log_transformation(dfs, cols_log: list):
    """
    Applies log1p transformation to specified columns in a list of DataFrames.
    """
    for df in dfs:
        for col in cols_log:
            if col in df.columns:
                df[col] = np.log1p(df[col])

def get_train_categories(df, col_name):
    """
    Extracts sorted unique categories from a training set column.
    """
    return sorted(list(set(df[col_name].tolist())))

def one_hot_encoding(df, col_name, categories, drop_first=True):
    """
    Performs one-hot encoding based on a fixed set of categories.
    """
    data = df[col_name].tolist()
    active_cats = categories[1:] if drop_first and len(categories) > 1 else categories
    
    encoded_mtx = []
    for item in data:
        row = [0] * len(active_cats)
        if item in active_cats:
            index = active_cats.index(item)
            row[index] = 1
        encoded_mtx.append(row)
    
    new_cols = [f'{col_name}_{cat}' for cat in active_cats]
    return pd.DataFrame(encoded_mtx, columns=new_cols, index=df.index)

def fit_scaler(df_train):
    """
    Learns min and range parameters for scaling.
    """
    params = {}
    for col in df_train.columns:   
        min_v = df_train[col].min()
        max_v = df_train[col].max()
        params[col] = (float(min_v), float(max_v - min_v))
    return params

def transform_scaler(df, params):
    """
    Applies scaling parameters to a DataFrame.
    """
    df_scaled = df.copy()
    for col, (min_v, data_range) in params.items():
        if col in df_scaled.columns:
            if data_range == 0:
                df_scaled[col] = 0.0
            else:
                df_scaled[col] = (df[col] - min_v) / data_range
    return df_scaled

# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================
class NeuralNetwork:
    def __init__(self, input_dim, h1=32, h2=16, h3=8, alpha=0.01):
        self.alpha = alpha
        # He INITIALIZATION
        self.w1 = np.random.randn(input_dim, h1) * np.sqrt(2/input_dim)
        self.w2 = np.random.randn(h1, h2) * np.sqrt(2/h1)
        self.w3 = np.random.randn(h2, h3) * np.sqrt(2/h2)
        self.w4 = np.random.randn(h3, 1) * np.sqrt(2/h3)
        self.b1 = np.zeros((1, h1))
        self.b2 = np.zeros((1, h2))
        self.b3 = np.zeros((1, h3))
        self.b4 = np.zeros((1, 1))
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def feedforward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = leaky_relu(self.z1, self.alpha)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = leaky_relu(self.z2, self.alpha)
        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = leaky_relu(self.z3, self.alpha)
        self.z4 = self.a3 @ self.w4 + self.b4
        self.a4 = sigmoid(self.z4)
        return self.a4

    def backpropagation(self, X, y, out):
        m = y.shape[0]
        dz4 = (out - y)
        dw4 = (self.a3.T @ dz4) / m
        db4 = np.sum(dz4, axis=0, keepdims=True) / m

        da3 = dz4 @ self.w4.T
        dz3 = da3 * derivative_leaky_relu(self.z3, self.alpha)
        dw3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = dz3 @ self.w3.T
        dz2 = da2 * derivative_leaky_relu(self.z2, self.alpha)
        dw2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.w2.T
        dz1 = da1 * derivative_leaky_relu(self.z1, self.alpha)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient Descent Update
        self.w1 -= self.alpha * dw1
        self.b1 -= self.alpha * db1
        self.w2 -= self.alpha * dw2
        self.b2 -= self.alpha * db2
        self.w3 -= self.alpha * dw3
        self.b3 -= self.alpha * db3
        self.w4 -= self.alpha * dw4
        self.b4 -= self.alpha * db4

    def predict(self, X):
        probs = self.feedforward(X.values if isinstance(X, pd.DataFrame) else X)
        return (probs > 0.5).astype(int).flatten(), probs.flatten()
