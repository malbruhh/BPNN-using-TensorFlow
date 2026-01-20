import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
import sys

# Add root to path for core import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bpnn_core

# CONFIGURATION
EPOCHS = 500
BATCH_SIZE = 32
PATIENCE = 100
ALPHA = 0.01

def detect_outliers_iqr(df, k=1.5):
    nums = df.select_dtypes(include='number')
    outlier_info = {}
    for c in nums.columns:
        if(nums[c].nunique() <= 2): continue
        q1, q3 = nums[c].quantile(0.25), nums[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask = (nums[c] < lower) | (nums[c] > upper)
        outlier_info[c] = {'count': int(mask.sum()), 'lower': lower, 'upper': upper}
    return outlier_info

def split_train_val(X, y, val_split=0.176, randomness=42):
    """
    Sub-splits the training source into Train and Validation.
    Note: 0.176 of 85% is roughly 15% of the original total.
    """
    np.random.seed(randomness)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    unique_classes = np.unique(y)
    train_idx, val_idx = [], []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        split = int(len(cls_indices) * val_split)
        val_idx.extend(cls_indices[:split])
        train_idx.extend(cls_indices[split:])
        
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

def train_model(nn, X_train, y_train, X_val, y_val):
    n_samples = X_train.shape[0]
    y_tr = np.array(y_train).reshape(-1, 1)
    y_v = np.array(y_val).reshape(-1, 1)
    
    best_loss = float('inf')
    p_count = 0
    init_a = nn.alpha

    for epoch in range(EPOCHS):
        if (epoch + 1) % 100 == 0:
            nn.alpha = init_a * (0.95 ** (epoch / 10))

        indices = np.random.permutation(n_samples)
        X_s, y_s = X_train.values[indices], y_tr[indices]

        for i in range(0, n_samples, BATCH_SIZE):
            bx, by = X_s[i:i+BATCH_SIZE], y_s[i:i+BATCH_SIZE]
            nn.backpropagation(bx, by, nn.feedforward(bx))

        t_out, v_out = nn.feedforward(X_train.values), nn.feedforward(X_val.values)
        t_loss, v_loss = bpnn_core.binary_cross_entropy(y_tr, t_out), bpnn_core.binary_cross_entropy(y_v, v_out)
        t_acc = np.mean((t_out > 0.5).astype(int) == y_tr) * 100
        v_acc = np.mean((v_out > 0.5).astype(int) == y_v) * 100

        nn.history['train_loss'].append(t_loss)
        nn.history['val_loss'].append(v_loss)
        nn.history['train_acc'].append(t_acc)
        nn.history['val_acc'].append(v_acc)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")

        if v_loss < best_loss:
            best_loss, p_count = v_loss, 0
        else:
            p_count += 1
        if p_count >= PATIENCE:
            print(f"Early stop at {epoch+1}")
            break
    return nn.history

def main():
    print("--- MODEL TRAINING STAGE (Using training_source.csv) ---")
    source_path = os.path.join('..', 'Dataset', 'training_source.csv')
    if not os.path.exists(source_path):
        print("Error: training_source.csv not found. Run isolation_split.py first.")
        return

    df = pd.read_csv(source_path)
    # Basic cleaning
    df = df.drop_duplicates()
    if 'age_group' in df.columns: df = df.drop(columns=['age_group'])

    X_train_full = df.drop(columns=['churn'])
    y_train_full = df['churn']

    # Sub-split into Train/Val
    X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)
    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}")

    # Preprocessing
    cols_log = ['seconds_of_use', 'frequency_of_use', 'frequency_of_sms', 
                'distinct_called_numbers', 'call_failure', 'customer_value', 'charge_amount']
    bpnn_core.log_transformation([X_train, X_val], cols_log)

    cat_cols = ['complains', 'tariff_plan', 'status']
    cat_info = []
    train_encoded, val_encoded = [], []
    for col in cat_cols:
        cats = bpnn_core.get_train_categories(X_train, col)
        cat_info.append({'col': col, 'categories': cats})
        train_encoded.append(bpnn_core.one_hot_encoding(X_train, col, cats))
        val_encoded.append(bpnn_core.one_hot_encoding(X_val, col, cats))
    
    X_train = X_train.drop(columns=cat_cols).join(train_encoded)
    X_val = X_val.drop(columns=cat_cols).join(val_encoded)

    scaler_params = bpnn_core.fit_scaler(X_train)
    X_train_s = bpnn_core.transform_scaler(X_train, scaler_params)
    X_val_s = bpnn_core.transform_scaler(X_val, scaler_params)

    # Train
    nn = bpnn_core.NeuralNetwork(input_dim=X_train_s.shape[1], alpha=ALPHA)
    history = train_model(nn, X_train_s, y_train, X_val_s, y_val)

    # Save Model Weights & Params for Deployment
    model_path = os.path.join('..', 'Deployment', 'model_v7.npz')
    config_path = os.path.join('..', 'Deployment', 'model_v7_config.json')
    
    np.savez(model_path, w1=nn.w1, w2=nn.w2, w3=nn.w3, w4=nn.w4, b1=nn.b1, b2=nn.b2, b3=nn.b3, b4=nn.b4)
    with open(config_path, 'w') as f:
        json.dump({'scale_params': scaler_params, 'cat_params': cat_info, 'cols_log': cols_log}, f)
    
    print(f"Model exported to Deployment/ folder.")

if __name__ == "__main__":
    main()