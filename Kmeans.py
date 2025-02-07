

import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:

    def __init__(self, K, n_iters):
        self.K = K # number of centroids
        self.n_iters = n_iters
        self.centroids = None
    

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.X = X
        # Initiate centroids
        c_id = np.random.choice(self.n_samples, self.K, replace = False)
        self.centroids = self.X[c_id]

        for _ in range(self.n_iters):

            clusters = self.create_cluster(self.X, self.centroids)

            updated_centroids = self.update_centroids(clusters)

            # check converge
            condition = np.sum(euclidean_distance(x1, x2) for x1, x2 in zip(updated_centroids, self.centroids)) == 0

            if condition:
                break
                
            
            self.centroids = updated_centroids
    

    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]

        for idx, x in enumerate(X):
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            assigned = np.argmin(distances)
            clusters[assigned].append(idx)
        
        return clusters
    

    def update_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for idc, cluster in enumerate(clusters):
            X_c = self.X[cluster]
            centroids[idc] = X_c.mean(axis = 0)
        
        return centroids
    

    def predict(self, X):
        return [self._predict(x) for x in X]
    

    def _predict(self, x):
        distances = [euclidean_distance(x, center) for center in self.centroids]
        return np.argmin(distances)
            



X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [8, 2], [10, 2],
    [9, 3]
])

# Instantiate the KMeans model
kmeans = KMeans(K=2, n_iters=100)

# Fit the model
kmeans.fit(X)

# Predict the clusters for the same dataset
predictions = kmeans.predict(X)

# Print out the predictions and centroids
print("Predictions:", predictions)
print("Centroids:", kmeans.centroids)