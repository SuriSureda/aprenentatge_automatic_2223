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
        Weights after fitting.
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
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):  # A cada iteració aconseguim una mostra i la seva classe
                update = self.eta * (target - self.predict(xi))  # Si la prediccio es correcta update = 0
                self.w_[1:] += update * xi # actualitzacio dels pesos
                self.w_[0] += update # actualitzacio del bias
                # Extra: calculam els errors de classificacio a cada iteració
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def __net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.__net_input(X) >= 0.0, 1, -1)
