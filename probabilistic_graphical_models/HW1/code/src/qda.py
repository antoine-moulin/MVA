import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import sigmoid


class QDA(BaseEstimator, ClassifierMixin):
    """
    Quadratic Discriminant Analysis maximum-likelihood estimator.

    p(y=1 | x) = 1 / (1 + exp(w^T x + b))
    """

    def __init__(self):
        """
        Initialization of an instance of QDA class.
        """
        self.n_features = 0
        self.n_samples = 0

        # maximum-likelihood estimators (MLE)
        self.pi = 0  # proportion of positive examples
        self.mu0 = np.zeros(self.n_features)  # empirical mean of the negative class
        self.mu1 = np.zeros(self.n_features)  # empirical mean of the positive class
        self.sigma0 = np.zeros((self.n_features, self.n_features))  # empirical covariance matrix of the negative class
        self.sigma1 = np.zeros((self.n_features, self.n_features))  # empirical covariance matrix of the positive class

        # parameters for the decision rule, obtained from the MLE
        self.w = np.zeros(self.n_features)
        self.b = 0

    def fit(self, X, y):
        """
        Fit our QDA model to the data set (X, y).

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
        self: QDA
               The QDA model trained on (X, y).
        """

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        neg_points = X[y == 0, :]
        pos_points = X[y == 1, :]

        # MLE
        self.pi = np.mean(y)
        self.mu0 = np.mean(neg_points, axis=0)
        self.mu1 = np.mean(pos_points, axis=0)
        self.sigma0 = ((neg_points - self.mu0).T @ (neg_points - self.mu0))/neg_points.shape[0]
        self.sigma1 = (pos_points - self.mu1).T @ (pos_points - self.mu1)/pos_points.shape[0]

        # parameters for the decision rule w and b
        inv_sigma0 = np.linalg.pinv(self.sigma0)
        inv_sigma1 = np.linalg.pinv(self.sigma1)

        self.w = inv_sigma1 @ self.mu1 - inv_sigma0 @ self.mu0
        self.b = np.log(self.pi / (1 - self.pi))\
            - (self.mu1.T @ inv_sigma1 @ self.mu1 - self.mu0.T @ inv_sigma0 @ self.mu0) / 2

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
        return sigmoid( -0.5*np.diag(X @ (np.linalg.inv(self.sigma1)-np.linalg.inv(self.sigma0)) @ X.T) + self.w.T @ X.T + self.b)

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
