

import numpy as np
from collections import Counter

class Node:
    def __init__(self, left=None, right=None, feature=None, threshold=None, value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features = n_features if not self.n_features else min(self.n_features, n_features)
        self.root = self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth=0):
        if X.shape[0] < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=self._most_common(y))
        
        feature_ids = np.random.choice(X.shape[1], self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_ids)

        if best_feature is None:
            return Node(value=self._most_common(y))

        left_id, right_id = self._split(X[:, best_feature], y, best_threshold)
        left = self._grow_tree(X[left_id, :], y[left_id], depth + 1)
        right = self._grow_tree(X[right_id, :], y[right_id], depth + 1)

        return Node(left, right, best_feature, best_threshold)

    def _most_common(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _best_split(self, X, y, fids):
        information_gain = float('-inf')
        split_feature, split_threshold = None, None

        for fid in fids:
            thresholds = np.unique(X[:, fid])
            for th in thresholds:
                current_gain = self.information_gain(X[:, fid], y, th)
                if current_gain > information_gain:
                    information_gain = current_gain
                    split_feature, split_threshold = fid, th

        return split_feature, split_threshold

    def _split(self, X_col, y, threshold):
        left = np.argwhere(X_col <= threshold).flatten()
        right = np.argwhere(X_col > threshold).flatten()
        return left, right

    def information_gain(self, X_col, y, threshold):
        parent = self._entropy(y)
        l, r = self._split(X_col, y, threshold)
        if len(l) == 0 or len(r) == 0:
            return 0
        return parent - (len(l) / len(y) * self._entropy(y[l]) + len(r) / len(y) * self._entropy(y[r]))

    def _entropy(self, y):
        ps = np.bincount(y, minlength=2)
        ps = ps / np.sum(ps)
        return -np.sum([p * np.log(p + 1e-9) for p in ps if p > 0])

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
y = np.array([1, 1, 0, 0])

tree = DecisionTree(max_depth=3)
tree.fit(X, y)

X_test = np.array([[1, 1], [0, 0]])
predictions = tree.predict(X_test)
print("Predictions:", predictions)