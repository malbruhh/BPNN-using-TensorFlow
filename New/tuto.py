# 1. Split	train_test_split	Create Train and Test sets.
# 2. Log	log_transformation	Fix skewness in numerical data.
# 3. Encode	one_hot_encoding	Convert Complains, Status, etc. to numbers.
# 4. Fit	get_scaling_params	Learn min/max from X_train only.
# 5. Scale	min_max_transform	Transform both X sets into 0.0 - 1.0 range.

import numpy as np
import pandas as pd
from tabulate import tabulate
def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def derivative_sigmoid(sigmoid_x):
    return sigmoid_x * (1- sigmoid_x )

def relu(x):
    return max(0,x)

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

def one_hot_encoding(df, col_name:str, drop_first = True):
    data = df[col_name].toList()
    categories = sorted(list(set(data)))
    
    #drop first column
    if drop_first and len(categories) > 1:
        categories = categories[1:]
        
    encoded_mtx = []
    #set all values to 0 first
    for item in data:
        row = [0] * len(categories)
        
        if item in categories:
            index = categories.index(item)
            row[index] = 1
        encoded_mtx.append(row)
    
    #rename column for one hot encoded column
    new_cols = [f'{col_name}_{category}' for category in categories]
    #convert back to dataframe
    converted_pd  = pd.DataFrame(encoded_mtx, columns=new_cols,index=df.index)
    print(f'[Changes] Applied one hot encoding for categorical columns')
    
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
            'indices': nums.index[mask].toList(),
            'lower': float(lower),
            'upper': float(upper)
        }
    return outlier_info

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
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=42)

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
    categorical = ['complains', 'tariff_plan', 'status']
    for col in categorical:
        train_encoded = one_hot_encoding(X_train, col, drop_first=True)
        X_train = X_train.drop(col, axis = 1).join(train_encoded)
        
        test_encoded = one_hot_encoding(X_test, col, drop_first=True)
        X_test = X_test.drop(col, axis = 1).join(test_encoded)
        
    #4. min-max scaling
    #fit on X_train only
    X_train_scale_params = get_scaling_params(X_train)
    X_train_scaled = min_max_transform(X_train, X_train_scale_params)
    X_test_scaled = min_max_transform(X_test, X_train_scale_params)
    
    