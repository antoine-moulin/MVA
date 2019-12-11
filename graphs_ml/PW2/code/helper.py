import matplotlib.pyplot as plt
import scipy
import numpy as np
import networkx as nx
import random
import scipy.io
import scipy.spatial.distance as sd
from scipy.sparse.csgraph import minimum_spanning_tree


def min_span_tree(W):
    """
    :param W: (n x n) adjacency matrix representing the graph
    :return: T: (n x n) matrix such that T[i,j] = True if the edge (i, j) is in the min spanning tree, and
                T[i, j] = False otherwise
    """
    tree = minimum_spanning_tree(W).toarray()
    T = tree != 0
    return T


def build_similarity_graph(X, var=1, eps=0, k=0):
    """
    Computes the similarity matrix for a given dataset of samples.
     
    :param X: (n x m) matrix of m-dimensional samples
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :return: W: (n x n) dimensional matrix representing the adjacency matrix of the graph
    """

    n = X.shape[0]
    W = np.zeros((n, n))

    # euclidean distance squared between points
    dists = sd.squareform(sd.pdist(X, "sqeuclidean"))

    """
    Build full (similarity) graph. The similarity function is s(x,y)=exp(-||x-y||^2/var) 
        similarities: (n x n) matrix with similarities between all possible couples of points.
    """
    similarities = np.exp(-dists / var)

    # If epsilon graph
    if k == 0:
        """
        compute an epsilon graph from the similarities             
        for each node x_i, an epsilon graph has weights             
        w_ij = d(x_i,x_j) when w_ij >= eps, and 0 otherwise          
        """
        W = similarities
        W[W < eps] = 0

    # If kNN graph
    if k != 0:
        """
        compute a k-nn graph from the similarities                   
        for each node x_i, a k-nn graph has weights                  
        w_ij = d(x_i,x_j) for the k closest nodes to x_i, and 0     
        for all the k-n remaining nodes                              
        Remember to remove self similarity and                       
        make the graph undirected                                    
        """
        sort = np.argsort(similarities)[:, ::-1]  # descending
        mask = sort[:, k + 1:]  # indices to mask
        for i, row in enumerate(mask):
            similarities[i, row] = 0
        np.fill_diagonal(similarities, 0)  # remove self similarity
        W = (similarities + similarities.T) / 2  # make the graph undirected

    return W


def build_laplacian(W, laplacian_normalization=""):
    """
    Compute Laplacian matrix.
    :param W:   (n x n) dimensional matrix representing the adjacency matrix of the graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """

    degree = W.sum(1)
    if not laplacian_normalization:
        return np.diag(degree) - W
    elif laplacian_normalization == "sym":
        aux = np.diag(1 / np.sqrt(degree))
        return np.eye(*W.shape) - aux.dot(W.dot(aux))
    elif laplacian_normalization == "rw":
        return np.eye(*W.shape) - np.diag(1 / degree).dot(W)
    else:
        raise ValueError


def plot_edges_and_points(X, Y, W, title=''):
    colors=['go-','ro-','co-','ko-','yo-','mo-']
    n=len(X)
    G=nx.from_numpy_matrix(W)
    nx.draw_networkx_edges(G,X)
    for i in range(n):
        plt.plot(X[i,0],X[i,1],colors[int(Y[i])])
    plt.title(title)
    plt.axis('equal')

            
def plot_graph_matrix(X, Y, W):
    plt.figure()
    plt.clf()
    plt.subplot(1, 2, 1)
    plot_edges_and_points(X,Y,W)
    plt.subplot(1, 2, 2)
    plt.imshow(W, extent=[0, 1, 0, 1])
    plt.show()           

            
def plot_clustering_result(X, Y, W, spectral_labels, kmeans_labels, normalized_switch=0):
    plt.figure()
    plt.clf()
    plt.subplot(1,3,1)
    plot_edges_and_points(X,Y,W,'ground truth')
    plt.subplot(1,3,2)
    if normalized_switch:
        plot_edges_and_points(X,spectral_labels,W,'unnormalized laplacian')
    else:
        plot_edges_and_points(X,spectral_labels,W,'spectral clustering')
    plt.subplot(1,3,3)
    if normalized_switch:
        plot_edges_and_points(X,kmeans_labels,W,'normalized laplacian')
    else:
        plot_edges_and_points(X,kmeans_labels,W,'k-means')
    plt.show()    
    
    
def plot_the_bend(X, Y, W, spectral_labels, eigenvalues_sorted):
    plt.figure()
    plt.clf()
    plt.subplot(1,3,1)
    plot_edges_and_points(X,Y,W,'ground truth')

    plt.subplot(1,3,2)
    plot_edges_and_points(X,spectral_labels,W,'spectral clustering')
    
    plt.subplot(1,3,3)
    plt.plot(np.arange(0,len(eigenvalues_sorted),1),eigenvalues_sorted,'v:')
    plt.show()


def plot_classification(X, Y, labels, var=1, eps=0, k=0, method='Hard HFS'):
    plt.figure()
    W = build_similarity_graph(X, var=var, eps=eps, k=k)

    plt.subplot(1, 2, 1)
    plot_edges_and_points(X, Y, W, 'Ground truth')

    plt.subplot(1, 2, 2)
    plot_edges_and_points(X, labels, W, method)
    plt.show()

    
def label_noise(Y, alpha):
    ind = np.arange(len(Y))
    random.shuffle(ind)
    Y[ind[:alpha]] = 3-Y[ind[:alpha]]
    return Y


def plot_classification_comparison(X, Y, hard_labels, soft_labels, var=1, eps=0, k=0):
    plt.figure()

    W = build_similarity_graph(X, var=var, eps=eps, k=k)

    plt.subplot(1,3,1)
    plot_edges_and_points(X, Y, W, 'ground truth')

    plt.subplot(1,3,2)
    plot_edges_and_points(X, hard_labels, W, 'Hard-HFS')

    plt.subplot(1,3,3)
    plot_edges_and_points(X, soft_labels, W, 'Soft-HFS')
    plt.show()


def plot_clusters(X, Y, fignum='data', show=False):
    """
    Function to plot clusters.

    :param X: (num_samples, 2) matrix of 2-dimensional samples
    :param Y:  (num_samples, ) vector of cluster assignment
    :param fignum: figure name
    :param show: If True, call plt.show()
    :return:
    """
    K = int(Y.max())
    plt.figure(fignum)
    for k in range(1, K+1):
        plt.plot(X[Y == k, 0], X[Y == k, 1], 'o')
    if show:
        plt.show()


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
