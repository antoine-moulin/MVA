import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LinearRegression(BaseEstimator, ClassifierMixin):
    """
    Linear Regression (Ordinary Least Squares).
    """

    def __init__(self):
        """
        Initialization of an instance of LinearRegression class.
        """
        self.n_features = 0
        self.n_samples = 0

        # Ordinary Least Squares solution
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        """
        Fit our linear regression model to the data set (X, y), i.e. compute the ordinary least squares estimator.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The training set.
            n_samples == number of points in the dataset.
            n_features == dimension of the points (e.g. each sample is in R^2).

        y: ndarray, shape (n_samples,)
            The class of each point (0: negative, 1: positive).
            n_samples == number of points in the dataset.

        Returns
        -------
        self: LinearRegression
               The LinearRegression model trained on (X, y).
        """

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # parameters for the decision rule
        X_tilde = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        theta = np.linalg.pinv(X_tilde.T @ X_tilde) @ X_tilde.T @ y

        self.b = theta[0]
        self.w = theta[1:]

        return self

    def predict(self, X):
        """
        Predict the classes of points in X.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The set of points whose classes are to predict.
            n_samples == number of points in the dataset.
            n_features == dimension of the points (e.g. each sample is in R^2).

        Returns
        -------
        predictions: ndarray, shape (n_samples)
                      The predicted classes.
                      n_samples == number of points in the dataset.
        """

        return self.decision_function(X) > 0.5

    def decision_function(self, X):
        """
        Return the value of the decision function of points in X.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The points on which to compute the values of the decision function.
            n_samples == number of points in the dataset.
            n_features == dimension of the points (e.g. each sample is in R^2).

        Returns
        -------
        df_values: ndarray, shape (n_samples,)
                    Values of the decision function obtained.
        """

        return self.w @ X.T + self.b

    def score(self, X, y):
        """
        Compute the accuracy of the classifier on the set X, provided the ground-truth y.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The set of points on which to compute the score.
            n_samples == number of points in the dataset.
            n_features == dimension of the points (e.g. each sample is in R^2).

        y: ndarray, shape (n_samples,)
            The ground-truth of the labels.
            n_samples == number of points in the dataset.

        Returns
        -------
        score: float
                Accuracy of the classifier.
        """

        return np.mean(self.predict(X) == y)
