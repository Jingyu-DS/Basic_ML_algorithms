

import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        self.X = X
        self.y = y

        n_sample, n_features = self.X.shape

        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)

        self.x_prior = np.zeros((self.n_classes, n_features))
        self.prior = np.zeros(self.n_classes)


        for cid, cl in enumerate(self.classes):
            X_c = self.X[self.y == cl]
            X_pos = X_c.sum(axis = 0)
            self.x_prior[cid] = X_pos 
            self.prior[cid] = X_c.shape[0]
        

        self.prior_prob = self.prior / len(self.y)
        self.x_prior_prob = (self.x_prior + 1) / (self.prior.reshape((-1, 1)) + n_features)
    

    def predict(self, X):
        return [self._predict(x) for x in X]
    

    def _predict(self, x):
        posteriors = []

        for cid, cl in enumerate(self.classes):
            prior = np.log(self.prior_prob[cid])
            posterior = np.sum(x * np.log(self.x_prior_prob[cid]))
            posterior += prior
        
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]



X = np.array([[1, 0, 1],
                  [1, 1, 0],
                  [0, 1, 1],
                  [0, 0, 1]])
# Labels (binary classification)
y = np.array([0, 0, 1, 1])

# Create and train Naive Bayes classifier
nb = NaiveBayes()
nb.fit(X, y)

# Test prediction on new data
X_test = np.array([[1, 0, 0],
                       [0, 1, 1]])

predictions = nb.predict(X_test)
print(predictions)  # Output: [0 1]