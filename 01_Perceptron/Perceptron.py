from array import array
from functools import reduce
import numpy as np

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        for _ in range(1,self.n_iter):
            for i in range(0,X.shape[0]):
                expected_y = self.output(X[i])
                self.updateWeights(X[i],expected_y, y[i])
        return self

    def updateWeights(self, X, expected_y, y):
        self.w_[0] += self.eta*(y - expected_y)
        for i in range(1, len(self.w_)):
            self.w_[i] += self.eta*(y - expected_y)*X[i-1]

    def output(self, X):
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return 1 if result >= 0 else -1

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """
        return list(map(self.output, X))
