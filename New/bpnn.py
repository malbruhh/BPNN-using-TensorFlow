class BPNN:
    def __init__(self, input_size, h1, h2, h3, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, h1) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2/h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, h3) * np.sqrt(2/h2)
        self.b3 = np.zeros((1, h3))
        self.W4 = np.random.randn(h3, output_size) * np.sqrt(2/h3)
        self.b4 = np.zeros((1, output_size))

# Activation functions
    def relu(self, x): return np.maximum(0, x)
    def relu_deriv(self, x): return (x > 0).astype(float)
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

# Forward propagation
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.relu(self.Z3)
        self.Z4 = self.A3 @ self.W4 + self.b4
        self.A4 = self.softmax(self.Z4)
        return self.A4

# Loss function
    def compute_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# Backward pass and weight updates
    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]

        dZ4 = y_pred - y_true
        dW4 = self.A3.T @ dZ4 / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m

        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self.relu_deriv(self.Z3)
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W4 -= self.lr * dW4; self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

#Predict function
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)