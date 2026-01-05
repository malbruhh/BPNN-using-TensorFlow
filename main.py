#Preprocessing Steps:
#1. Standardization: Rename columns and convert to lowercase.

#2. Deduplication: Remove exact duplicate rows.

#3. Feature Selection: Drop redundant age_group.

#4. Encoding: Map categorical 1/2 values to 0/1.

#5. Data Splitting: 80/20 Stratified Split.

#6. Outlier Treatment: Log Transform or Winsorize usage data.

#7. Normalization: Min-Max Scaling for all features.

#handling Outliers After IQR:
#1) Log Transformation for:
# Seconds of use
# Frequency of use
# Frequency of sms
# Distinct Called numbers
# Call Failure
# Customer Value
# Charge Amount

#2) Keep the outliers as it is because they represent real-world scenarios:
# Can use min-max scaling for standardization 
# Age
# Subscription Length

#3) Drop column (Redundancy):
# Age group

#4) One hot encoding for categorical data:
# Complain
# Tariff Plan
# Status

#Network Architecture:
# Input Layer: 12 neurons (features)
# Hidden Layer: 8 neurons, ReLU activation
# Output Layer: 1 neuron, Sigmoid activation

#--0 Import library --
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # Turns off oneDNN warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Hides Info and Warning logs (0=all, 1=warn, 2=err, 3=none)

from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.metrics import Recall, Precision
import seaborn as sns

EPOCHS = 50
BATCH_SIZE = 32

def read_file(file_path):
    df = pd.read_csv(file_path)
    print(f'Loaded {file_path}, shape={df.shape}\n')
    
    #convert column name in better formatting
    new_columns = []
    for col in df.columns:
        clean_name = col.strip()
        clean_name = clean_name.replace('  ', ' ')
        clean_name = clean_name.replace(' ', '_')
        clean_name = clean_name.lower()
        new_columns.append(clean_name)  

    df.columns = new_columns
    df.info()
    return df

def detect_outliers_iqr(df, k=1.5):
    nums = df.select_dtypes(include='number')
    outlier_info = {}
    
    for c in nums.columns:
        #skip binary columns
        if(nums[c].nunique() <= 2):
            continue
        
        q1 = nums[c].quantile(0.25) #1st quartile
        q3 = nums[c].quantile(0.75) #3rd quartile
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask = (nums[c] < lower) | (nums[c] > upper)
        outlier_info[c] = {
            'count': int(mask.sum()),
            'indices': nums.index[mask].tolist(),
            'lower': float(lower),
            'upper': float(upper)
        }
    return outlier_info

#-- 1st Model
# Optimizer: Adam - adaptive learning rate algorithm
# Loss Function: Mean Squared Error (MSE)
def NN_adam_MSE(X_train_final, y_train_final, X_test_final, y_test_final):
    # Input Layer: 12 neurons (features)
    # Hidden Layer: 8 neurons, ReLU
    # Output Layer: 1 neuron, Sigmoid 
    model = Sequential()
    model.add(Input(shape=(12,)))
    model.add(Dense(units=8, activation='relu', name='hidden_layer'))
    model.add(Dense(units=1, activation='sigmoid', name='output_layer'))
    
    model.summary()
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy', Recall(), Precision()])
        
    history = model.fit(
        X_train_final, 
        y_train_final, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, # Uses 20% to check overfitting
        verbose=1
    )
    loss, accuracy, precision, recall= model.evaluate(X_test_final, y_test_final)    
    y_hat = model.predict(X_test_final)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat] # <0.5 = not churn, >=0.5 = churn
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return history, y_hat

#-- 2nd Model
# Optimizer: Stochastic Gradient Descent (SGD) - constant learning rate: 0.01
# Loss Function: Binary Cross Entropy (BCE)
def NN_sgd_BCE(X_train_final, y_train_final, X_test_final, y_test_final):
    # Input Layer: 12 neurons (features)
    # Hidden Layer: 8 neurons, ReLU
    # Output Layer: 1 neuron, Sigmoid 
    model = Sequential()
    model.add(Input(shape=(12,)))
    model.add(Dense(units=8, activation='relu', name='hidden_layer'))
    model.add(Dense(units=1, activation='sigmoid', name='output_layer'))
    
    model.summary()
    model.compile(optimizer = 'sgd', loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])
        
    history = model.fit(
        X_train_final, 
        y_train_final, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, # Uses 20% to check overfitting
        verbose=1
    )
    loss, accuracy, precision, recall= model.evaluate(X_test_final, y_test_final)    
    y_hat = model.predict(X_test_final)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat] # <0.5 = not churn, >=0.5 = churn
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return history, y_hat

#-- 3rd Model
# 2 hidden layers
# Optimizer: Adam - adaptive learning rate algorithm
# Loss Function: Mean Squared Error (MSE)
def NN_adam_MSE_extra_layer(X_train_final, y_train_final, X_test_final, y_test_final):
    # Input Layer: 12 neurons (features)
    # Hidden Layer: 16 neurons, ReLU
    # Hidden Layer: 8 neurons, ReLU
    # Output Layer: 1 neuron, Sigmoid 
    model = Sequential()
    model.add(Input(shape=(12,)))
    model.add(Dense(units=16, activation='relu', name='first_hidden_layer'))
    model.add(Dense(units=8, activation='relu', name='second_hidden_layer'))
    model.add(Dense(units=1, activation='sigmoid', name='output_layer'))
    
    model.summary()
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy', Recall(), Precision()])
        
    history = model.fit(
        X_train_final, 
        y_train_final, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, # Uses 20% to check overfitting
        verbose=1
    )
    loss, accuracy, precision, recall= model.evaluate(X_test_final, y_test_final)    
    y_hat = model.predict(X_test_final)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat] # <0.5 = not churn, >=0.5 = churn
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return history, y_hat

#--Graph: Loss Curve--
def evaluate_model_performance(history, y_true, y_pred, title):
    # Create a figure with 3 subplots in 1 row
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Model Evaluation: {title}', fontsize=16, fontweight='bold')

    # --- 1. Loss Curve Plot ---
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    axes[0].plot(epochs, train_loss, label='Training Loss', color='blue', lw=1.5, marker='x', ms=4)
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='orange', lw=1.5, marker='x', ms=4)
    axes[0].set_title('Loss Curve (Learning Curve)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- 2. Accuracy Plot ---
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    axes[1].plot(epochs, train_acc, label='Training Accuracy', color='blue', lw=1.5, marker='x', ms=4)
    axes[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange', lw=1.5, marker='x', ms=4)
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- 3. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stay', 'Churn'], 
                yticklabels=['Stay', 'Churn'], 
                ax=axes[2])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_ylabel('Actual')
    axes[2].set_xlabel('Predicted')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    #--1 Load Data--
    df = read_file("Dataset/Customer Churn.csv")
    outliers = detect_outliers_iqr(df)
    
    #--2 Outlier Detection and Data Splitting--
    table_data = []
    print('\nOutlier summary (IQR method):')
    for col, info in outliers.items():
        if info['count'] > 0:
        # Calculate percentage
            perc = (info["count"] / len(df)) * 100
            
            # Add a list (row) to our table_data
            table_data.append([
                col, 
                info["count"], 
                f"{info['lower']:.3f}", 
                f"{info['upper']:.3f}", 
                f"{perc:.2f}%"
            ])
    headers = ["Column", "Outlier Count", "Lower Bound", "Upper Bound", "Percentage"]
    print(tabulate(table_data, headers=headers))
    
    #--Data Preprocessing Steps--
    #remove duplicate rows
    df = df.drop_duplicates()
    print(f'\n[Changes] Removed duplicate rows. New shape={df.shape}\n')
    
    #remove age_group column due to redundancy
    df = df.drop(columns=['age_group'])
    print(f'\n[Changes] Dropped column: age_group due to redundancy. New shape={df.shape}\n\n')

    #apply one-hot encoding to categorical columns
    df = pd.get_dummies(df, columns=['complains', 'tariff_plan', 'status'], drop_first=True)
    print(f'[Changes] Applied one-hot encoding to categorical columns: complains, tariff_plan, status. New shape={df.shape}\n\n')

    #Split data into train and test set
    X = df.drop(columns=['churn'], axis=1)
    Y = df['churn']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=42)
    
    #--3 Outlier Treatment--
    #Log Transformation (Both Train and Test set)--
    print(f'''Using Log Transformation for outlier treatment on columns:
          - seconds_of_use
          - frequency_of_use
          - frequency_of_sms
          - distinct_called_numbers
          - call_failure
          - customer_value
          - charge_amount\n
          ''')
    cols_to_log = [
        'seconds_of_use',
        'frequency_of_use',
        'frequency_of_sms',
        'distinct_called_numbers',
        'call_failure',
        'customer_value',
        'charge_amount'
    ]
    
    #log transformation seperate train and test set to avoid data leakage
    for col in cols_to_log:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])
    print(f'[Changes] Applied log transformation to selected columns. Current shape: {df.shape}\n')
    
    scaler = MinMaxScaler()
    scaler.fit(X_train) #scaled the data based on X train
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test) #use X_train scaler for unseen data
    
    #convert back to dataframe
    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    y_train_final = y_train.values.reshape(-1, 1)
    y_test_final = y_test.values.reshape(-1, 1)
    
    #--4 1st Model: Adam Optimizer, MSE Loss --
    print('\n--- 1st Model: Using Mean Squared Error (MSE)')
    first_history, first_y_hat = NN_adam_MSE(X_train_final, y_train_final, X_test_final, y_test_final) 
    evaluate_model_performance(first_history, y_test_final, first_y_hat, title='MSE and Adam Optimizer')
    
    input("Press Enter to continue to the second model...")
    
    #--6 2nd Model: SGD Optimizer, Binary Cross Entropy Loss --
    print('\n--- 2nd Model: Using Binary Cross Entropy Loss Function (BCE)')
    second_history, second_y_hat = NN_sgd_BCE(X_train_final, y_train_final, X_test_final, y_test_final)
    evaluate_model_performance(second_history, y_test_final, second_y_hat, title='BCE and SGD Optimizer')
    
    input("Press Enter to continue to the third model...")

    #--8 3rd Model: Adam Optimizer, MSE loss, 2 hidden layer
    print('\n--- 3nd Model: Using Mean Squared Error With extra Hidden Layer')
    third_history, third_y_hat = NN_adam_MSE_extra_layer(X_train_final, y_train_final, X_test_final, y_test_final)
    evaluate_model_performance(third_history, y_test_final, third_y_hat, title='MSE and Adam Optimizer (2 Hidden Layers)')
    
    #--10 Compare between 3 models
    print('\n--- Comparison of 3 Models ---')
    model_names = ['MSE with Adam', 'BCE with SGD', 'MSE with Adam (2 Hidden Layers)']
    histories = [first_history, second_history, third_history]
    y_hats = [first_y_hat, second_y_hat, third_y_hat]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

    plt.figure(figsize=(16, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        plt.plot(history.history['loss'], color=colors[i], label=f'{name} (Train)', lw=1.5)
        plt.plot(history.history['val_loss'], color=colors[i], linestyle='--', label=f'{name} (Val)', lw=1.5)

    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small', ncol=2) 
    plt.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        plt.plot(history.history['accuracy'], color=colors[i], label=f'{name} (Train)', lw=1.5)
        plt.plot(history.history['val_accuracy'], color=colors[i], linestyle='--', label=f'{name} (Val)', lw=1.5)

    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
