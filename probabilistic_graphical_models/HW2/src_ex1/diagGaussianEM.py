import numpy as np
from sklearn.cluster import k_means
from scipy.stats import multivariate_normal as MVN

def diagGaussianEM(X, K, max_iters = 50):
    """
    Gaussian Mixture model with diagonal covariance matrices.

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_features)
        The set of datapoints.
        n_samples == number of points in the dataset.
        n_features == n_dimension of the points (e.g. each sample is in R^2).

    K: int
        The number of classes.
        
    Returns
    -------
    p: float
            Accuracy of the classifier.
    mus: float
            Accuracy of the classifier.
    sigmas: float
            Accuracy of the classifier.
    labels: float
            Accuracy of the classifier.
    """
        
    n_samples = X.shape[0]
    n_dim = X.shape[1]
    
    # Intialization : using kmeans++
    mus, labels, _ = k_means(X, K, init='k-means++')
        
    P = np.empty((K, 1)) 
    for k in range(K):
        P[k, 0] = np.sum(labels == k) / n_samples
    
    sigmas = np.stack([np.diagflat(100*np.ones(n_dim)) for k in range(K)])
    diag_idx = np.diag_indices(n_dim)
    q = np.empty((K, n_samples))
    

    
    for iter in range(max_iters):

        # E step
        
        condProba = np.stack([MVN.pdf(X, mean=mus[k], cov=sigmas[k], allow_singular = True) for k in range(K)]) # p(X|k)
        
        q[:] = P * condProba
        q[:] /= np.sum(q, axis = 0, keepdims=True)
        w = np.sum(q, axis = 1, keepdims=True)  # weights
        
        # M step
        
        # update probabilities
        P[:, 0] = np.mean(q, axis = 1)
        
        #update parameters
        for k in range(K):
            mus[k][:] = X.T @ q[k] / w[k]
            diagos_ = (X - mus[k]) ** 2
            qk = q[k, :, None] 
            new_sigma = np.sum(qk * diagos_, axis=0) / w[k]
            sigmas[k][diag_idx] = new_sigma
    

    labels = q.argmax(axis = 0) 
    
    return P, mus, sigmas, labels

