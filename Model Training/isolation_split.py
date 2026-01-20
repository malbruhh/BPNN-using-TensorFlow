import numpy as np
import pandas as pd
import os
import sys

# Ensure root is in path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bpnn_core

def isolate_test_data(raw_csv_path, test_split=0.15, randomness=42):
    """
    Strictly carves out a test set from the raw data.
    Saves 'test_data.csv' and 'training_source.csv' in the Dataset directory.
    """
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"--- ISOlATION STEP: Carving out Test Set ---")
    df = pd.read_csv(raw_csv_path)
    # Normalization (needed for stratification)
    df.columns = [col.strip().replace('  ', ' ').replace(' ', '_').lower() for col in df.columns]
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Stratified Split
    np.random.seed(randomness)
    unique_classes = np.unique(y)
    test_indices = []
    source_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        
        split_point = int(len(cls_indices) * test_split)
        test_indices.extend(cls_indices[:split_point])
        source_indices.extend(cls_indices[split_point:])
        
    np.random.shuffle(test_indices)
    np.random.shuffle(source_indices)
    
    df_test = df.iloc[test_indices]
    df_source = df.iloc[source_indices]
    
    test_path = os.path.join(dataset_dir, 'test_data.csv')
    source_path = os.path.join(dataset_dir, 'training_source.csv')
    
    df_test.to_csv(test_path, index=False)
    df_source.to_csv(source_path, index=False)
    
    print(f"Isolated Test Set Reserved: {len(df_test)} rows -> {test_path}")
    print(f"Training Source Prepared:   {len(df_source)} rows -> {source_path}")
    print(f"--- Isolation Complete ---")

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Customer Churn.csv')
    if os.path.exists(raw_path):
        isolate_test_data(raw_path)
    else:
        print(f"Error: Original dataset not found at {raw_path}")
