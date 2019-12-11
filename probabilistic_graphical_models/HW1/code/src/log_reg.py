import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import sigmoid


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistical Regression maximum-likelihood estimator.
    """

    def __init__(self):
        self.n_features = 0
        self.n_samples = 0

        # parameters for the decision rule
        self.w = np.zeros((1, self.n_features + 1))  # the intercept is included in w
        self.nb_iter = 1
        
    def gradLogit(self, X, y):
        """
        Computes the gradient of the log-likelihood for the Logistic Regression model
    
        Parameters
        ----------
        w: ndarray, shape(1, n_features)
        
        X: ndarray, shape (n_samples, n_features)
            The training set.
            n_samples == number of points in the dataset.
            n_features == dimension of the points (e.g. each sample is in R^2).
    
        y: ndarray, shape (n_samples,)
            The class of each point (0: negative, 1: positive).
            n_samples == number of points in the dataset.
    
        Returns
        -------
        gradLL: ndarray, shape (1, n_features)
                The gradient of the log-likelihood
        """
        gradLL = np.zeros((1,X.shape[1]))
        for k in range(X.shape[0]):
            gradLL[0,:] += np.reshape((y[k] - sigmoid(self.w@np.reshape(X[k,:],(X.shape[1],1))))*X[k,:],(-1,))

        return gradLL
        
    def hessLogit(self, X, y):
        """
        Computes the Hessian of the log-likelihood for the Logistic Regression model
    
        Parameters
        ----------
        w: ndarray, shape(1, n_features)
            
        X: ndarray, shape (n_samples, n_features)
            The training set.
            n_samples == number of points in the dataset.
            n_features == dimension of the points (e.g. each sample is in R^2).
    
        y: ndarray, shape (n_samples,)
            The class of each point (0: negative, 1: positive).
            n_samples == number of points in the dataset.
    
        Returns
        -------
        hessLL: ndarray, shape (n_features, n_features)
                The Hessian of the log-likelihood
        """
        diag = np.array([sigmoid(self.w@np.reshape(X[k,:],(X.shape[1],1)))*sigmoid(-self.w@np.reshape(X[k,:],(X.shape[1],1))) for k in range (X.shape[0]) ])
        diag = np.diag(diag.flatten())
        return - X.T @ diag @ X

    def fit(self, X, y, nb_max_iter = 200, eps = 1e-3):
        """
        Fit our Logistic Regression model to the data set (X, y)

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
        self: logReg
               The Logistic Regression model trained on (X, y).
        """
        
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        X = np.append(X, np.ones((self.n_samples,1)), axis=1)
        self.w = np.zeros((1, self.n_features + 1))

        cond = np.inf  # start with cond greater than eps (assumption)
        
        while cond > eps and self.nb_iter < nb_max_iter:
            w_old = self.w
            self.w = w_old - self.gradLogit(X, y) @ np.linalg.pinv(self.hessLogit(X, y))
            cond = np.linalg.norm(self.w - w_old)
            self.nb_iter += 1
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
        X = np.append(X, np.ones((X.shape[0],1)), axis=1)
        return np.reshape(sigmoid(X @ self.w.T) > 0.5,(-1,))

    def decision_function(self, X):
        X = np.append(X, np.ones((X.shape[0],1)), axis=1)
        return np.reshape(sigmoid(X @ self.w.T),(-1,))

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
