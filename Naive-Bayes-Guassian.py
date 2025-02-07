


import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = self.X.shape 

        # classes
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)


        # mean var prior

        self.mean = np.zeros((self.n_classes, n_features))
        self.var = np.zeros((self.n_classes, n_features))
        self.priors = np.zeros(self.n_classes)

        for cid, cl in enumerate(self.classes):
            X_c = self.X[self.y == cl]
            self.mean[cid] = X_c.mean(axis = 0)
            self.var[cid] = X_c.var(axis = 0)
            self.priors[cid] = X_c.shape[0] / len(self.y)
    


    def predict(self, X):
        return [self._predict(x) for x in X]
    

    def _predict(self, x):
        posteriors = []

        for cid, cl in enumerate(self.classes):
            prior = np.log(self.priors[cid])
            posterior = np.sum(np.log(self._pdf(x, cid)))
            posterior += prior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    

    def _pdf(self, x, c):
        c_mean = self.mean[c]
        c_var = self.var[c]
        numerator = np.exp(-(x - c_mean) ** 2 / 2 * c_var)
        denominator = np.sqrt(2 * np.pi * c_var)
        return numerator / denominator



X_train = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [4, 3], [3, 4], [4, 4]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Instantiate the NaiveBayes classifier
nb = NaiveBayes()

# Fit the model
nb.fit(X_train, y_train)

# Predict with new data
X_test = np.array([[1.5, 1.5], [3.5, 3.5]])
predictions = nb.predict(X_test)

print("Predictions:", predictions)