
import numpy as np

class SVM:
    def __init__(self, lambda_param, lr, n_iters):
        self.lambda_param = lambda_param
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                condition = y[idx] * (np.dot(x, self.weights) - self.bias) >= 1

                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x, y[idx]))
                    self.bias -= self.lr * y[idx]
    

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias  ## something needs attention: minus bias rather than plus bias
        return np.sign(approx)



# Create a simple linearly separable dataset
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [1, 0], [0, 1], [0, 0]])
y = np.array([1, 1, 1, -1, -1, -1, -1, -1])

# Instantiate the SVM model
svm = SVM(lr=0.01, lambda_param=0.01, n_iters=1000)

# Train the SVM model
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)

# Print the results
print("Predictions:", predictions)
print("Weights:", svm.weights)
print("Bias:", svm.bias)
