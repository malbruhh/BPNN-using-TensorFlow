import numpy as np
import pandas as pd
import json
import os
import sys

# Add root to path for core import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bpnn_core

def run_deployment_test(test_data_path, model_path, config_path):
    print("--- DEPLOYMENT STAGE: Evaluating on Isolated Test Set ---")
    
    if not os.path.exists(test_data_path):
        print("Error: test_data.csv not found.")
        return

    # Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load Model
    weights = np.load(model_path)
    
    # Load Data
    df = pd.read_csv(test_data_path)
    # Basic Cleaning to match training columns
    df.columns = [col.strip().replace('  ', ' ').replace(' ', '_').lower() for col in df.columns]
    if 'age_group' in df.columns: df = df.drop(columns=['age_group'])
    
    X = df.drop(columns=['churn'])
    y_true = df['churn'].values

    # Preprocessing (USING SAME CORE FUNCTIONS)
    bpnn_core.log_transformation([X], config['cols_log'])
    
    encoded_parts = []
    for p in config['cat_params']:
        encoded_parts.append(bpnn_core.one_hot_encoding(X, p['col'], p['categories']))
    
    X = X.drop(columns=[p['col'] for p in config['cat_params']]).join(encoded_parts)
    X_scaled = bpnn_core.transform_scaler(X, config['scale_params'])

    # Initialize NN with loaded weights
    nn = bpnn_core.NeuralNetwork(input_dim=X_scaled.shape[1])
    nn.w1, nn.w2, nn.w3, nn.w4 = weights['w1'], weights['w2'], weights['w3'], weights['w4']
    nn.b1, nn.b2, nn.b3, nn.b4 = weights['b1'], weights['b2'], weights['b3'], weights['b4']

    # Predict
    preds, probs = nn.predict(X_scaled)
    
    accuracy = np.mean(preds == y_true) * 100
    print(f"\nRESULTS ON ISOLATED TEST DATA:")
    print(f"Total Test Samples: {len(y_true)}")
    print(f"Accuracy:           {accuracy:.2f}%")
    
    # Save predictions
    df['predicted_churn'] = preds
    df['churn_probability'] = probs
    df.to_csv(os.path.join(os.path.dirname(test_data_path), 'final_predictions.csv'), index=False)
    print(f"Detailed results saved to Dataset/final_predictions.csv")

if __name__ == "__main__":
    t_path = os.path.join('..', 'Dataset', 'test_data.csv')
    m_path = 'model_v7.npz'
    c_path = 'model_v7_config.json'
    run_deployment_test(t_path, m_path, c_path)
