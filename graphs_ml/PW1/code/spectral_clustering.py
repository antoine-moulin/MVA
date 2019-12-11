import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy

from utils import plot_clustering_result, plot_the_bend
from build_similarity_graph import build_similarity_graph
from generate_data import blobs, two_moons, point_and_circle

from sklearn.metrics import pairwise_distances
from utils import min_span_tree


def build_laplacian(W, laplacian_normalization="rw"):
    """
    Compute graph Laplacian.

    :param W: adjacency matrix
    :param laplacian_normalization:  string selecting which version of the laplacian matrix to construct
                                     'unn':  unnormalized,
                                     'sym': symmetric normalization
                                     'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """

    n = W.shape[0]
    L = np.zeros((n, n))
    D = np.diag(W @ np.ones(n))

    if laplacian_normalization == 'unn':
        L = D - W
    elif laplacian_normalization == 'sym':
        L = np.eye(n) - np.linalg.inv(D)**0.5 @ W @ np.linalg.inv(D)**0.5
    elif laplacian_normalization == 'rw':
        L = np.eye(n) - np.linalg.inv(D) @ W

    return L


def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """

    eigenvalues, eigenvectors = scipy.linalg.eig(L)
    index = np.argsort(eigenvalues) # for ordering
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    U = eigenvectors.real  # because scipy.linalg.eig returns complex vectors with a null imaginary part
    U = U[:, chosen_eig_indices]

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]
    """
    # In the case of two clusters, cf questions 2.1. and 2.2., one can use the sign of the second eigenvector to compute
    # the cluster assignments. In order to have something more general, it is k-means that is used here.
    Y = KMeans(num_classes).fit_predict(U)
    return Y


def two_blobs_clustering():
    """
    TO BE COMPLETED

    Clustering of two blobs. Used in questions 2.1 and 2.2
    """

    question = '2.2'

    # Get data and compute number of classes
    X, Y = blobs(600, n_blobs=2, blob_var=0.15, surplus=0)
    num_classes = len(np.unique(Y))
    n = X.shape[0]

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'rw'

    if question == '2.1':
        # as the graph has to be connected in this question, we construct a epsilon-graph using a MST
        dists = pairwise_distances(X).reshape((n, n))  # dists[i, j] = euclidean distance between x_i and x_j
        min_tree = min_span_tree(dists)
        distance_threshold = np.max(dists[min_tree])
        eps = np.exp(- distance_threshold**2 / (2*var))

        # choice of eigenvectors to use
        chosen_eig_indices = [1]  # indices of the ordered eigenvalues to pick

        # build similarity graph and laplacian
        W = build_similarity_graph(X, var=var, eps=eps)
        L = build_laplacian(W, laplacian_normalization)

    elif question == '2.2':
        # choice of eigenvectors to use
        chosen_eig_indices = [0, 1]

        # choice of k for the k-nn graph
        k = 20

        # build similarity graph and laplacian
        W = build_similarity_graph(X, var=var, k=k)
        L = build_laplacian(W, laplacian_normalization)

    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    # Plot results
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def choose_eigenvalues(eigenvalues):
    """
    Function to choose the indices of which eigenvalues to use for clustering.

    :param eigenvalues: sorted eigenvalues (in ascending order)
    :return: indices of the eigenvalues to use
    """
    # eigengaps = np.diff(eigenvalues)  # first idea

    eigengaps = np.zeros(len(eigenvalues)-1)

    # we start to compare the eigengaps from the first non-zero eigenvalue
    epsilon = 1e-9
    if np.argwhere(eigenvalues > epsilon).size > 0:
        first_index = np.argwhere(eigenvalues > epsilon)[0][0]
    else:
        first_index = 0

    for k in range(first_index, len(eigenvalues)-1):
        eigengaps[k] = (eigenvalues[k+1] - eigenvalues[k]) / (eigenvalues[k+1] + eigenvalues[k] + epsilon)  # second idea
        # eigengaps[k] = (eigenvalues[k+1] - eigenvalues[k]) / (sum(eigenvalues[:k+1]) + epsilon)  # third idea

    eig_ind = np.arange(np.argmax(eigengaps) + 1)

    return eig_ind


def spectral_clustering_adaptive(L, num_classes=2):
    """
    Spectral clustering that adaptively chooses which eigenvalues to use.
    :param L: Graph Laplacian (standard or normalized)
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    eigenvalues, eigenvectors = scipy.linalg.eig(L)
    index = np.argsort(eigenvalues)  # for ordering
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    U = eigenvectors.real  # because scipy.linalg.eig returns complex vectors with a null imaginary part
    U = U[:, choose_eigenvalues(eigenvalues.real)]

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]
    """
    Y = KMeans(num_classes).fit_predict(U)
    return Y


def find_the_bend():
    """
    TO BE COMPLETED

    Used in question 2.3
    :return:
    """

    # the number of samples to generate
    num_samples = 600

    # Generate blobs and compute number of clusters
    # var_blobs = 0.03  # question 2.3.
    var_blobs = 0.2  # question 2.4.
    X, Y = blobs(num_samples, 4, var_blobs)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 20
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'rw'  # 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization

    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

    """
    compute first 15 eigenvalues and call choose_eigenvalues() to choose which ones to use. 
    """
    eigenvalues, _ = scipy.linalg.eig(L)
    eigenvalues = eigenvalues[np.argsort(eigenvalues)].real
    eigenvalues = eigenvalues[:15]
    chosen_eig_indices = choose_eigenvalues(eigenvalues)  # indices of the ordered eigenvalues to pick

    """
    compute spectral clustering solution using a non-adaptive method first, and an adaptive one after (see handout) 
    Y_rec = (n x 1) cluster assignments [0,1,..., c-1]    
    """
    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    # Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes)

    plot_the_bend(X, Y, L, Y_rec, eigenvalues)


def two_moons_clustering():
    """
    TO BE COMPLETED.

    Used in question 2.7
    """
    # Generate data and compute number of clusters
    X, Y = two_moons(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    eps = 0.8
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'rw'

    # build laplacian
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    L = build_laplacian(W, laplacian_normalization)

    # spectral clustering
    Y_rec = spectral_clustering_adaptive(L, num_classes=num_classes)

    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def point_and_circle_clustering():
    """
    TO BE COMPLETED.

    Used in question 2.8
    """
    # Generate data and compute number of clusters
    X, Y = point_and_circle(600, sigma=.2)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    eps = 0.4
    var = 1  # exponential_euclidean's sigma^2

    # build laplacian
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    L_unn = build_laplacian(W, 'unn')
    L_norm = build_laplacian(W, 'rw')

    Y_unn = spectral_clustering_adaptive(L_unn, num_classes=num_classes)
    Y_norm = spectral_clustering_adaptive(L_norm, num_classes=num_classes)

    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1)


def parameter_sensitivity():
    """
    TO BE COMPLETED.

    A function to test spectral clustering sensitivity to parameter choice.

    Used in question 2.9
    """
    # the number of samples to generate
    num_samples = 500

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'rw'
    # chosen_eig_indices = [0, 1]

    """
    Choose candidate parameters
    """
    # the number of neighbours for the graph or the epsilon threshold
    # parameter_candidate = np.arange(3, 33, 3)
    parameter_candidate = np.linspace(0.2, 1, 9)
    parameter_performance = []

    for param in parameter_candidate:
        # Generate data
        X, Y = two_moons(num_samples)
        num_classes = len(np.unique(Y))

        W = build_similarity_graph(X, eps=param)
        # W = build_similarity_graph(X, k=param)
        L = build_laplacian(W, laplacian_normalization)

        Y_rec = spectral_clustering_adaptive(L, num_classes)
        # Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)

        parameter_performance += [skm.adjusted_rand_score(Y, Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    plt.show()


if __name__ == '__main__':
    two_blobs_clustering()
    find_the_bend()
    two_moons_clustering()
    point_and_circle_clustering()
    parameter_sensitivity()
