import numpy as np
from collections import Counter

def distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,*,k=3):
        self.k=3
        self.X_train=None
        self.y_train=None

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X):
        predictions = [self._predict(x)for x in X]
        return predictions

    def _predict(self,x):
        distances=[distance(x,self.X_train)for i in range(self.X_train)]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common=Counter(k_labels).most_common()
        return most_common

