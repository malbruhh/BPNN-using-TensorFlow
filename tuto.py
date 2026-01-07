import numpy as np
import csv

#how machine learning work in pyhton:
#   4                                   4                            1 (0 or 1)
#[INPUT] --*-- [w1] -- [activation + HIDDEN] --*--[w2]--[activation + OUTPUT]
# 1100x4        4x4                  1100x4       4x1                 1100x1
#
# follow linear algebra matrices formula: 
# Supposed matrices size : AxB * YxZ, the size will be A*Z
# size of weight will be determined by using the size from layer forward.

# variables:
# learning rate: too small= slow; too big = jumps way too much
# threshold = accuracy to determine stability ; system need > 70% accuracy to consider stable
# epoch = 1 epoch : 1 iteration every input.

#this activation function will use for converting to 0-1 value
#derivative = how much to adjust weight during backpropagation
def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)
class NeuralNetwork:
    def __init__(self, inp, exp):
        self.input = inp
        # self.exp = exp.reshape(-1,1)
        self.exp = exp
        self.weight1 = np.random.rand(4,4) #in matrices form
        self.weight2 = np.random.rand(4,1) #in matrices form
        self.alpha = 0.01
        
    def feedforward(self):
        self.hidden = sigmoid(np.dot(self.input, self.weight1)) #multiply with w1 matrices
        self.output = sigmoid(np.dot(self.hidden, self.weight2)) #multiple with w2 matrices
        
    def backpropagation(self):
        error = self.exp - self.output #this is simple, we need to use advanced (MSE)

        #1 change hidden weight first
        # matrices.T = transpose = swap row and column
        delta_output = error * derivative_sigmoid(self.output) #tells the machine how much weight to adjust
        adjust_weight2 = self.alpha * np.dot(self.hidden.T, delta_output)
        
        #adjust input weight using chain rule(input weight is determined by delta_output and hidden weight)
        delta_input = derivative_sigmoid(self.hidden) * np.dot(delta_output, self.weight2.T)
        adjust_weight1 = self.alpha * np.dot(self.input.T, delta_input)
        
        self.weight1 += adjust_weight1
        self.weight2 += adjust_weight2
        
#data will be seperated into 2 set:
#[Train] and [Test]
def main():
    with open('BankNote_Authentication.txt', mode = 'r') as f:
        lines = f.readlines()
        #skip header
        # next(lines)

    #randomized dataset to avoid ordered set
    np.random.shuffle(lines) 
    
    #lines right now is in string value rather than numbers
    #change stringified line into float number
    input= []
    expected =[]
    #seperate between input and expected_output data
    for line in lines:
        line = line.split(',')
        input.append([float(x) for x in line[0:4]]) #insert into input data
        expected.append([float(line[4])])

    #convert into numpy array to use other numpy func
    input = np.array(input)
    expected = np.array(expected)
    
    #split data into training(80%) and test(20%)
    
    input_train = input[:1100]
    expected_train = expected[:1100]
    
    input_test = input[1100:]
    expected_test = expected[1100:]
    
    #------- start BPNN ------------
    nn = NeuralNetwork(input_train,expected_train)
    
    #default modifier
    initial_alpha = 0.01
    total_epochs = 5000
    threshold = 0.01
    
    for epoch in range(1, total_epochs + 1):
        
        # 1. Update the alpha learning decay
        nn.alpha = initial_alpha * (1 / (1 + 0.01 * epoch))
        
        # --- STEP 1: TRAIN ---
        nn.input = input_train # Ensure we are using training data
        nn.feedforward()
        nn.backpropagation()
        
        # Calculate Training Accuracy
        train_diff = np.abs(nn.output - expected_train)
        train_acc = (train_diff[train_diff <= threshold].size / train_diff.size) * 100

        # --- STEP 2: TEST (Every 100 epochs) ---
        if epoch % 100 == 0:
            # We temporarily swap to test data JUST to check performance
            nn.input = input_test
            nn.feedforward() 
            
            test_diff = np.abs(nn.output - expected_test)
            test_acc = (test_diff[test_diff <= threshold].size / test_diff.size) * 100
            
            print(f"Epoch {epoch} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    
    # --- FINAL TESTING RESULTS ---
    # 1. Switch to test data
    nn.input = input_test
    nn.feedforward()

    # 2. Convert probabilities to 0 or 1 using 0.5 as the decision boundary
    # .flatten() ensures we are comparing 1D arrays
    predictions = (nn.output > 0.5).astype(int).flatten()
    actuals = expected_test.astype(int).flatten()

    # 3. Calculate TP, TN, FP, FN
    tp = np.sum((predictions == 1) & (actuals == 1))
    tn = np.sum((predictions == 0) & (actuals == 0))
    fp = np.sum((predictions == 1) & (actuals == 0))
    fn = np.sum((predictions == 0) & (actuals == 1))

    # 4. Print the "Confusion Matrix"
    print("\n" + "="*30)
    print("FINAL TEST CONFUSION MATRIX")
    print("="*30)
    print(f"True Positives (Correct Real):  {tp}")
    print(f"True Negatives (Correct Fake):  {tn}")
    print(f"False Positives (Type I Error): {fp}")
    print(f"False Negatives (Type II Error):{fn}")
    print("-"*30)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100
    print(f"FINAL TEST ACCURACY: {accuracy:.2f}%")
    #seperate to avoid overwriting actual training data
if __name__ == '__main__':
    main()