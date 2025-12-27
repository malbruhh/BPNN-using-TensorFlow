# create 2-3 graph
#1 Loss curve(learning curve)-MSE,binary cross entropy
#2 accuracy over time(epoch)
#3 confusion matrix heatmap
#4 Gradient/Weight distribution(Histogram)


#preprocess data
#one-hot encoding (for nominal)
#Feature scaling
#class imbalance
#data split

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
        
    #--4 Model Training: Fit, Predict and Evaluate--
    EPOCHS = 500
    BATCH_SIZE = 32
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
    model.compile(optimizer = 'sgd', learning_rate=0.01, loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])
        
    #--4 Model Training: Fit, Predict and Evaluate--
    EPOCHS = 500
    BATCH_SIZE = 32
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
def loss_curve_plot(history):
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Training Loss' color='blue')
    plt.plot(val_loss, label='Validation Loss'color='orange')
    plt.title('Model Loss (Learning Curve)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0,100)
    plt.legend()
    plt.grid(True)
    plt.show()

def accuracy_plot(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy', color='blue') # 'bo-' blue dots and lines
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy', color='orange') # 'ro-' red dots and lines
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def confustion_mtx_map(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stay', 'Churn'], 
            yticklabels=['Stay', 'Churn'])
    plt.title('Confusion Matrix: Churn Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
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
    
    #-- 1st Model: MSE Loss --
    print('\n--- 1st Model: Using Mean Squared Error (MSE)')
    first_history, first_y_hat = NN_adam_MSE(X_train_final, y_train_final, X_test_final, y_test_final)
    
    #--5 Plotting Graphs--
    loss_curve_plot(first_history) #loss curve graph
    accuracy_plot(first_history) #accuracy over epochs plot
    confustion_mtx_map(y_test_final, first_y_hat) #Confusion Matrix Heatmap
    
    
if __name__ == "__main__":
    main()
