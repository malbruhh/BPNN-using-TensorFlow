import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

EPOCHS = 500
BATCH_SIZE = 32
PATIENCE = 100
ALPHA = 0.01
def sigmoid(x): return 1 /(1 + np.exp(-x))
def leaky_relu(x, alpha=ALPHA): return np.where(x > 0, x, alpha * x) # If x > 0, return x. If x <= 0, return alpha * x
def derivative_leaky_relu(Z, alpha =ALPHA):
    dZ = np.ones_like(Z)
    dZ[Z <= 0] = alpha
    return dZ
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def read_file(file_path: str):
    df = pd.read_csv(file_path)
    new_columns = [col.strip().replace('  ', ' ').replace(' ', '_').lower() for col in df.columns]
    df.columns = new_columns
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
def split_data(X, y, test_split=0.2, randomness=None):
    # Set seed for reproducibility
    if randomness is not None:
        np.random.seed(randomness)
    
    # reset X and Y current index
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
def log_transformation(df_train, df_test, cols_log: list):
    for col in cols_log:
        df_train[col] = np.log1p(df_train[col])
        df_test[col] = np.log1p(df_test[col])
    print(f'[Changes] Applied log transformation to selected columns.')
    return df_train, df_test
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
    return converted_pd
def get_scaling_params(df_train):
    param_dict = {}
    for col in df_train.columns:   
        min_val = min(df_train[col])
        max_val = max(df_train[col])
        param_dict[col] = (min_val, max_val - min_val)
    return param_dict
def min_max_transform(df, params: dict):
    df_scaled = df.copy()

    for col, (min_v, data_range) in params.items():
        if data_range == 0:
            df_scaled[col] = 0.0 #base case float number
        else:
            df_scaled[col] = (df[col] - min_v) / data_range
            
    return df_scaled
def evaluate_model_performance(history:dict, y_true:list, y_pred:list, title:str):
    # Create a figure within 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(6, 20))
    fig.suptitle(f'Model Evaluation: {title}', fontsize=16, fontweight='bold')

    # --- 1. Loss Curve Plot ---
    train_loss = history['train_loss']
    val_loss = history['test_loss']
    epochs = range(1, len(train_loss) + 1)

    axes[0].plot(epochs, train_loss, label='Training Loss', color='blue', lw=1.5, marker='x', ms=4)
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='orange', lw=1.5, marker='x', ms=4)
    axes[0].set_title('Loss Curve (Learning Curve)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- 2. Accuracy Plot ---
    train_acc = history['train_acc']
    val_acc = history['test_acc']
    axes[1].plot(epochs, train_acc, label='Training Accuracy', color='blue', lw=1.5, marker='x', ms=4)
    axes[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange', lw=1.5, marker='x', ms=4)
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- 3. Confusion Matrix ---
    actual = np.array(y_true).flatten()
    predicted = np.array(y_pred).flatten()
    
    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stay', 'Churn'], 
            yticklabels=['Stay', 'Churn'],
            ax=axes[2])
    axes[2].set_title('Confusion Matrix: Churn Prediction')
    axes[2].set_ylabel('Actual')
    axes[2].set_xlabel('Predicted')
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

class NeuralNetwork:
    def __init__(self, input_dimension, hidden1_nodes=32, hidden2_nodes=16, hidden3_nodes=8, alpha=ALPHA):
        #Step 0: Initialization
        self.alpha = alpha
        self.weight1 = np.random.randn(input_dimension, hidden1_nodes) * np.sqrt(2/input_dimension) #He Initializaion keeps Weight stable
        self.weight2 = np.random.randn(hidden1_nodes, hidden2_nodes) * np.sqrt(2/hidden1_nodes)
        self.weight3 = np.random.randn(hidden2_nodes, hidden3_nodes) * np.sqrt(2/hidden2_nodes)
        self.weight4 = np.random.randn(hidden3_nodes, 1) * np.sqrt(2/hidden3_nodes)
        self.bias1 = np.zeros((1, hidden1_nodes))
        self.bias2 = np.zeros((1, hidden2_nodes))
        self.bias3 = np.zeros((1, hidden3_nodes))
        self.bias4 = np.zeros((1, 1))
        
        self.train_loss, self.test_loss, self.train_acc, self.test_acc = [], [], [], []
        self.history = {
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'train_acc': self.train_acc,
            'test_acc': self.test_acc
        }    
        
    def feedforward(self,X):
        # Step 1 : Calc Hidden Layer
        self.hidden1_Z = X @ self.weight1 + self.bias1
        self.hidden1_A = leaky_relu(self.hidden1_Z, self.alpha)
        self.hidden2_Z = self.hidden1_A @ self.weight2 + self.bias2
        self.hidden2_A = leaky_relu(self.hidden2_Z, self.alpha)
        self.hidden3_Z = self.hidden2_A @ self.weight3 + self.bias3
        self.hidden3_A = leaky_relu(self.hidden3_Z,self.alpha)
        
        # Step 2: Calc Output Layer
        self.output_Z = self.hidden3_A @ self.weight4 + self.bias4
        self.output_A= sigmoid(self.output_Z)
        
        return self.output_A
    
    def backpropagation(self, X, y, output):
        size = y.shape[0] 
        
        # Step 3: Calculate Error
        d_output = (output - y)
        
        # Step 4: Calculate Output Error Gradient
        d_weight4 = self.hidden3_A.T @ d_output / size
        d_bias4 = np.sum(d_output, axis=0, keepdims=True) / size

        # Step 5: Calculate Hidden Error Gradient
        d_hidden3_A = d_output @ self.weight4.T
        d_hidden3_Z = d_hidden3_A * derivative_leaky_relu(self.hidden3_Z, self.alpha)
        d_weight3 = self.hidden2_A.T @ d_hidden3_Z / size
        d_bias3 = np.sum(d_hidden3_Z, axis=0, keepdims=True) /size
        
        d_hidden2_A = d_hidden3_Z @ self.weight3.T
        d_hidden2_Z = d_hidden2_A * derivative_leaky_relu(self.hidden2_Z,self.alpha)
        d_weight2 = self.hidden1_A.T @ d_hidden2_Z / size
        d_bias2 = np.sum(d_hidden2_Z, axis=0, keepdims=True) / size
        
        d_hidden1_A = d_hidden2_Z @ self.weight2.T
        d_hidden1_Z = d_hidden1_A * derivative_leaky_relu(self.hidden1_Z, self.alpha)
        d_weight1 = X.T @ d_hidden1_Z / size
        d_bias1 = np.sum(d_hidden1_Z, axis=0, keepdims=True) / size
        
        # Step 6: Update Output Weight
        self.weight4 -= self.alpha * d_weight4
        self.bias4 -= self.alpha * d_bias4
        
        # Step 7 and 8: Update Hidden Weight
        self.weight3 -= self.alpha * d_weight3
        self.bias3 -= self.alpha * d_bias3
        self.weight2 -= self.alpha * d_weight2
        self.bias2 -= self.alpha * d_bias2
        self.weight1 -= self.alpha * d_weight1
        self.bias1 -= self.alpha * d_bias1

    def calculate_accuracy(self, y_true, y_pred_prob):
        # Threshold at 0.5 for binary classification
        predictions = (y_pred_prob > 0.5).astype(int)
        correct = np.sum(predictions == y_true)
        return correct / len(y_true)
    
    def train(self, X_train, y_train, X_test, y_test):
        epochs = EPOCHS
        batch_size = BATCH_SIZE
        max_error = 0.01
        best_loss = float('inf')
        patience_count = 0
        patience = PATIENCE
        initial_alpha = self.alpha
        
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_tr = np.array(y_train).reshape(-1, 1)
        X_te = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_te = np.array(y_test).reshape(-1, 1)
        
        n_samples = X_tr.shape[0]
        
        for epoch in range(epochs):
            
            if (epoch + 1) % 100 == 0:
                self.alpha = initial_alpha * (0.95 ** (epoch / 10)) 
                print(f"Learning rate reduced to: {self.alpha:.6f}")
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_tr_shuffled = X_tr[indices]
            y_tr_shuffled = y_tr[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_tr_shuffled[i:i+batch_size]
                batch_y = y_tr_shuffled[i:i+batch_size]
                
                output = self.feedforward(batch_X)
                self.backpropagation(batch_X, batch_y, output)
            
            # Calculate metrics on full dataset
            output_full = self.feedforward(X_tr)
            train_loss = binary_cross_entropy(y_tr, output_full)
            train_acc = np.mean((output_full > 0.5).astype(int) == y_tr) * 100
            
            output_test = self.feedforward(X_te)
            test_loss = binary_cross_entropy(y_te, output_test)
            test_acc = np.mean((output_test > 0.5).astype(int) == y_te) * 100
            
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            
            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_count = 0
            else:
                patience_count += 1
            
            if patience_count >= patience:
                print(f'[Training Stopped] Patience {patience} reached')
                break
            
            if train_loss <= max_error:
                print(f'[Training Stopped] Max error {max_error} reached')
                break
            
        return self.history

    def predict(self, X):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X        
        h1_Z = X_vals @ self.weight1 + self.bias1
        h1_A = leaky_relu(h1_Z)
        h2_Z = h1_A @ self.weight2 + self.bias2
        h2_A = leaky_relu(h2_Z)
        h3_Z = h2_A @ self.weight3 + self.bias3
        h3_A = leaky_relu(h3_Z)
        o_Z = h3_A @ self.weight4 + self.bias4
        probs = sigmoid(o_Z)
        return (probs > 0.5).astype(int).flatten()

def main():

    path = r'..\Dataset\Customer Churn.csv'
    df = read_file(path)
    df.head(20)
    df.info()
    df.describe()

    outliers = detect_outliers_iqr(df)
    table_data = []
    print('\nOutlier summary (IQR method):')
    for col, info in outliers.items():
        if info['count'] > 0:
        # Calculate percentage
            perc = (info["count"] / len(df)) * 100
            
            # Add a list (row) to our table_data
            table_data.append([col, info["count"], f"{info['lower']:.3f}",
                f"{info['upper']:.3f}", f"{perc:.2f}%"])
    headers = ["Column", "Outlier Count", "Lower Bound", "Upper Bound", "Percentage"]
    print(tabulate(table_data, headers=headers))
        
    df = df.drop_duplicates()
    print(f'\n[Changes] Removed duplicate rows. New shape={df.shape}\n')
    df = df.drop(columns=['age_group'])
    print(f'\n[Changes] Dropped column: age_group due to redundancy. New shape={df.shape}\n\n')


    X = df.drop(columns=['churn'], axis=1)
    Y = df['churn']
    X_train,X_test,y_train,y_test = split_data(X,Y,test_split=0.2, randomness=42)
    print(f'[Changes] Successfully split data into Training and Testing.')


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

    train_encoded_parts = []
    test_encoded_parts = []

    categorical = ['complains', 'tariff_plan', 'status']
    for col in categorical:
        train_categories = get_train_categories(X_train, col)
        train_encoded_parts.append(one_hot_encoding(X_train, col, train_categories, drop_first=True))        
        test_encoded_parts.append(one_hot_encoding(X_test, col, train_categories, drop_first=True))
    print(f'[Changes] Applied one hot encoding to categorical columns')

    #drop old column and join new columns
    X_train = X_train.drop(columns=categorical).join(train_encoded_parts)
    X_test = X_test.drop(columns=categorical).join(test_encoded_parts)



    X_train_scale_params = get_scaling_params(X_train)
    X_train_scaled = min_max_transform(X_train, X_train_scale_params)
    X_test_scaled = min_max_transform(X_test, X_train_scale_params)
    print(f'[Changes] Applied Min Max Scaler on numerical columns')

    input_dim = X_train_scaled.shape[1]


    nn = NeuralNetwork(input_dimension= input_dim, alpha=ALPHA)
    history = nn.train(X_train_scaled, y_train, X_test_scaled, y_test)    
    y_hat = nn.predict(X_test_scaled)

    print('Model finished training and validation.')

    evaluate_model_performance(history, y_test, y_hat, title='Model 1: Binary Cross Entropy')
    
if __name__ == '__main__':
    main()
