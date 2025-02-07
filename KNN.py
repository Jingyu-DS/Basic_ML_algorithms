

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:

    def __init__(self, K):
        self.K = K
    

    def fit(self, X, y):
        self.X = X
        self.y = y
    

    def predict(self, X):
        return [self._predict(x) for x in X]
    

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X]
        neighbor_id = np.argsort(distances)[:self.K]
        labels = self.y[neighbor_id]
        return Counter(labels).most_common(1)[0][0]



        
X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1])  # Labels
print(X_train.shape)

# New points for prediction
X_test = np.array([[5, 5], [1, 1]])
# Initialize KNN classifier with k=3
knn = KNN(3)

# Train the model
knn.fit(X_train, y_train)

# Predict the labels for the test set
predictions = knn.predict(X_test)

print("Predictions:", predictions)