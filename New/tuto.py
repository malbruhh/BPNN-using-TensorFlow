# 1. Split	train_test_split	Create Train and Test sets.
# 2. Log	log_transformation	Fix skewness in numerical data.
# 3. Encode	one_hot_encoding	Convert Complains, Status, etc. to numbers.
# 4. Fit	get_scaling_params	Learn min/max from X_train only.
# 5. Scale	min_max_transform	Transform both X sets into 0.0 - 1.0 range.
# 6. Balance SMOTE              Apply SMOTE on train data

import numpy as np
import pandas as pd
from tabulate import tabulate
from imblearn.over_sampling import SMOTE

EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def derivative_sigmoid(sigmoid_x):
    return sigmoid_x * (1- sigmoid_x )

def relu(x):
    return max(0,x)

def derivative_relu(relu_x):
    return (relu_x>0).astype(float)

def split_data(X, Y, test_split=0.2, randomness=None):
    # Set seed for reproducibility
    if randomness is not None:
        np.random.seed(randomness)
    
    # Ensure X and y are reset/aligned indices for safety
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Identify unique classes and their indices (0 and 1)
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        # Get indices of rows belonging to this class
        cls_indices = np.where(y == cls)[0]
        
        # Shuffle indices within this specific class
        np.random.shuffle(cls_indices)

        # Determine the split point
        total_count = len(cls_indices)
        test_count = int(total_count * test_split)
        
        # Split indices
        cls_test = cls_indices[:test_count]
        cls_train = cls_indices[test_count:]
        
        # Add to main lists
        test_indices.extend(cls_test)
        train_indices.extend(cls_train)
        
    # Shuffle the final combined indices so they aren't grouped by class
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Use .iloc for DataFrames to select the rows
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test
    
#get scaling parameter on X_TRAIN for minmax scaler
#eg: train_min_val, train_data_range = get_scaling_param(X_TRAIN)
def get_scaling_params(df_train):
    param_dict = {}
    for col in df_train.columns:   
        min_val = min(df_train[col])
        max_val = max(df_train[col])
        param_dict[col] = (min_val, max_val - min_val)
    return param_dict

#eg: min_max_transform(X_TEST, train_min_val, train_data_range)
def min_max_transform(df, params: dict):
    df_scaled = df.copy()

    for col, (min_v, data_range) in params.items():
        if data_range == 0:
            df_scaled[col] = 0.0 #base case float number
        else:
            df_scaled[col] = (df[col] - min_v) / data_range
            
    return df_scaled

#get categories for train data only
def get_train_categories(df, col_name):
    return sorted(list(set(df[col_name].tolist())))

def one_hot_encoding(df, col_name:str, categories, drop_first = True):
    data = df[col_name].tolist()
    
    active_cats = categories[1:] if drop_first and len(categories) > 1 else categories
    
    encoded_mtx = []
    for item in data:
        row = [0] * len(active_cats)
        if item in active_cats:
            index = active_cats.index(item)
            row[index] = 1
        encoded_mtx.append(row)
    
    #rename column for one hot encoded column
    new_cols = [f'{col_name}_{cat}' for cat in active_cats]
    #convert back to dataframe
    converted_pd  = pd.DataFrame(encoded_mtx, columns=new_cols,index=df.index)
    print(f'[Changes] Applied one hot encoding to categorical columns')
    return converted_pd

def log_transformation(df_train, df_test, cols_log: list):
    for col in cols_log:
        df_train[col] = np.log1p(df_train[col])
        df_test[col] = np.log1p(df_test[col])
    print(f'[Changes] Applied log transformation to selected columns.')
    return df_train, df_test

def calculate_adaptive_alpha(weight):
    pass


def read_file(file_path: str):
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

class NeuralNetwork:
    def __init__(self, input_dimension, hidden_nodes = 8, alpha = 0.01):
        self.alpha = alpha
        self.weight1 = np.random.rand(input_dimension,hidden_nodes) * np.sqrt(2/input_dimension) #dim = inputX8 for hidden using He Initialization
        self.weight2 = np.random.rand(hidden_nodes, 1) * np.sqrt(2/hidden_nodes) #dim = 8x1 for output
        
        #[0]=Train Loss, [1]=Test Loss, [2]=Train Acc, [3]=Test Acc
        self.history = [[], [], [], []]
    
    def feedforward(self,X):
        #Input -> Hidden
        self.hidden_input = np.dot(X, self.weight1)
        self.hidden_output = relu(self.hidden_input)
        #Hidden -> Output
        self.output_input = np.dot(self.hidden_output, self.weight2)
        self.output = sigmoid(self.output_input)
        return self.output
    
    def backpropagation(self, X, y, output):
        error = y - output
        #calc delta(Output: sigmoid , Hidden: ReLU)
        delta_output = error * derivative_sigmoid(output)        
        delta_hidden = np.dot(delta_output, self.weight2.T) * derivative_relu(self.hidden_output)
        
        #update weights
        self.weight2 += self.alpha * np.dot(self.hidden_output.T, delta_output)
        self.weight1 += self.alpha * np.dot(X.T, delta_hidden)
    
    def calculate_accuracy(self, y_true, y_pred_prob):
        # Threshold at 0.5 for binary classification
        predictions = (y_pred_prob > 0.5).astype(int)
        correct = np.sum(predictions == y_true)
        return correct / len(y_true)
    
    def train(self, X_train, y_train,X_test, y_test):
        epochs = EPOCHS
        max_error = 0.01 # achieve best accuracy
        best_loss = float('inf')
        patience_count = 0
        patience = PATIENCE
        #make sure data in numpy
        X_tr, y_tr = X_train.values, y_train
        X_te, y_te = X_test.values, y_test
        
        for epoch in range(epochs):
            output = self.feedforward(X_tr)
            self.backpropagation(X_tr, y_tr, output)
            
            #training metrics
            train_loss = np.mean(np.square(y_tr - output)) #Mean Squared Error
            train_acc = np.mean((output > 0.5).astype(int) == y_tr) * 100 #return as list of 0 and 1, then compare with length y_tr 
            
            #Validation
            output_test = self.feedforward(X_te)
            test_loss = np.mean(np.square(y_te - output_test))
            test_acc = np.mean((output_test > 0.5).astype(int) == y_te) * 100

            self.history[0].append(train_loss)
            self.history[1].append(test_loss)
            self.history[2].append(train_acc)
            self.history[3].append(test_acc)
            
            #print current state
            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

            #stopping condition
            #reach close to zero error 
            if train_loss < max_error:
                print(f'[Training Stopped] Max error {max_error} reached')
                break
            
            #does not learn(overfit)
            if test_loss < best_loss:
                best_loss = test_loss
                patience_count = 0
            else:
                patience_count += 1
            patient_count +=1 if test_loss < best_loss else 0
            if patience_count >= patience:
                print(f'[Training Stopped] Patience {patience} reached')
                break
            
        return self.history
    
    def predict(self, X):
        # Ensure input is numpy array
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        # Simple forward pass
        h_input = np.dot(X_vals, self.weight1)
        h_output = relu(h_input)
        o_input = np.dot(h_output, self.weight2)
        probs = sigmoid(o_input)
        # Return 0 or 1
        return (probs > 0.5).astype(int).flatten()

def main():
    #--1 Load Data--
    df = read_file("Dataset/Customer Churn.csv")
    outliers = detect_outliers_iqr(df)
    
    #--2 Outlier Detection--
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
    
    #--2. Data Preprocessing--
    #remove duplicate rows
    df = df.drop_duplicates()
    print(f'\n[Changes] Removed duplicate rows. New shape={df.shape}\n')
    
    #remove age_group column due to redundancy
    df = df.drop(columns=['age_group'])
    print(f'\n[Changes] Dropped column: age_group due to redundancy. New shape={df.shape}\n\n')

    # need to remove this
    X = df.drop(columns=['churn'], axis=1)
    Y = df['churn']
    
    
    #1. split
    X_train,X_test,y_train,y_test = split_data(X,Y,test_size=0.2, randomness=42)

    #2. Log transform
    cols_to_log = [
        'seconds_of_use',
        'frequency_of_use',
        'frequency_of_sms',
        'distinct_called_numbers',
        'call_failure',
        'customer_value',
        'charge_amount'
    ]
    X_train, X_test = log_transformation(X_train,X_test,cols_to_log)
    
    #3. one hot encoding
    train_encoded_parts = []
    test_encoded_parts = []
    
    categorical = ['complains', 'tariff_plan', 'status']
    for col in categorical:
        train_categories = get_train_categories(X_train, col)
        train_encoded_parts.append(one_hot_encoding(X_train, col, train_categories, drop_first=True))        
        test_encoded_parts.append(one_hot_encoding(X_test, col, train_categories, drop_first=True))
        
    X_train = X_train.drop(columns=categorical).join(train_encoded_parts)
    X_test = X_test.drop(columns=categorical).join(test_encoded_parts)
    
    #4. min-max scaling
    #fit on X_train only
    X_train_scale_params = get_scaling_params(X_train)
    X_train_scaled = min_max_transform(X_train, X_train_scale_params)
    X_test_scaled = min_max_transform(X_test, X_train_scale_params)
    
    smote = SMOTE(random_state=42)
    
    X_train_final, y_train_final = smote.fit_resample(X_train_final, y_train_final)
    
    #reshape all into dataframe
    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    y_train_final = y_train.values.reshape(-1, 1)
    y_test_final = y_test.values.reshape(-1, 1)
    