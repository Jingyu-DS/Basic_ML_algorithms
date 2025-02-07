
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None 
        self.mean = None
    

    def fit(self, X):

        # PCA requires th data to be mean-centered. 
        self.mean = X.mean(axis = 0)
        X = X - self.mean

        # cov_X = X.T X
        cov_X = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov_X) # remember this

        # print(eigenvalues)
        # print(eigenvectors)

        # eigenvectors = eigenvectors.T # to faciliate the selection 
        top_n = np.argsort(eigenvalues)[::-1][:self.n_components]  # argsort is used: easy to get it wrong
        self.components = eigenvectors[:, top_n]  
    


    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components) # T based on previous



np.random.seed(42)
X = np.random.randn(10, 3)  # 10 samples, 3 features (3D points)

# Instantiate PCA with 2 components (reduce 3D to 2D)
pca = PCA(n_components=2)

# Fit the PCA model on the dataset
pca.fit(X)

# Transform the dataset to the new 2D space
X_transformed = pca.transform(X)
print(X_transformed.shape)

# Display the original and transformed datasets
print("Original 3D data:\n", X)
print("\nTransformed 2D data:\n", X_transformed)