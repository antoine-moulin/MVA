import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import sigmoid


class LDA(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis maximum-likelihood estimator.

    p(y=1 | x) = 1 / (1 + exp(w^T x + b))
    """

    def __init__(self):
        """
        Initialization of an instance of LDA class.
        """
        self.n_features = 0
        self.n_samples = 0

        # maximum-likelihood estimators (MLE)
        self.pi = 0  # proportion of positive examples
        self.mu0 = np.zeros(self.n_features)  # empirical mean of the negative class
        self.mu1 = np.zeros(self.n_features)  # empirical mean of the positive class
        self.sigma = np.zeros((self.n_features, self.n_features))  # empirical covariance matrix

        # parameters for the decision rule, obtained from the MLE
        self.w = np.zeros(self.n_features)
        self.b = 0

    def fit(self, X, y):
        """
        Fit our LDA model to the data set (X, y).

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
        self: LDA
               The LDA model trained on (X, y).
        """

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        neg_points = X[y == 0, :]
        pos_points = X[y == 1, :]

        # MLE
        self.pi = np.mean(y)
        self.mu0 = np.mean(neg_points, axis=0)
        self.mu1 = np.mean(pos_points, axis=0)
        self.sigma = (neg_points - self.mu0).T @ (neg_points - self.mu0)\
            + (pos_points - self.mu1).T @ (pos_points - self.mu1)
        self.sigma /= self.n_samples

        # parameters for the decision rule w and b
        inv_sigma = np.linalg.pinv(self.sigma)

        self.w = inv_sigma @ (self.mu1 - self.mu0)
        self.b = np.log(self.pi / (1 - self.pi))\
            - (self.mu1.T @ inv_sigma @ self.mu1 - self.mu0.T @ inv_sigma @ self.mu0) / 2

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
        predictions: ndarray, shape (n_samples,)
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

        return sigmoid(self.w.T @ X.T + self.b)

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
