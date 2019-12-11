"""
Functions to build and visualize similarity graphs, and to choose epsilon in epsilon-graphs.
"""

import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os
from sklearn.metrics import pairwise_distances
from utils import plot_clusters

from utils import plot_graph_matrix, min_span_tree
from generate_data import worst_case_blob, blobs, two_moons, point_and_circle


def similarity(x, y, var=1.0):
    """
    Computes the similarity function for two samples x and y, defined by:
        d(x, y) = exp(||x - y||^2 / (2 * var))

    :param x: (m, ) a sample
    :param y: (m, ) a sample
    :param var: the sigma value for the exponential function, already squared
    :return:
        a real number, representing the similarity between x and y
    """

    return np.exp(- np.linalg.norm(x - y)**2 / (2*var))


def build_similarity_graph(X, var=1.0, eps=0.0, k=0):
    """
    TO BE COMPLETED.

    Computes the similarity matrix for a given dataset of samples. If k=0, builds epsilon graph. Otherwise, builds
    kNN graph.

    :param X:    (n x m) matrix of m-dimensional samples
    :param var:  the sigma value for the exponential function, already squared
    :param eps:  threshold eps for epsilon graphs
    :param k:    the number of neighbours k for k-nn. If zero, use epsilon-graph
    :return:
        W: (n x n) dimensional matrix representing the adjacency matrix of the graph
    """
    n = X.shape[0]
    W = np.zeros((n, n))

    """
    Build similarity graph, before threshold or kNN
    similarities: (n x n) matrix with similarities between all possible couples of points.
    The similarity function is d(x,y)=exp(-||x-y||^2/(2*var))
    """
  
    similarities = np.zeros((n, n))  # this matrix is symmetric
    for i in range(n):
        for j in range(i):
            similarities[i, j] = similarity(X[i, :], X[j, :], var)
            similarities[j, i] = similarities[i, j]

    # If epsilon graph
    if k == 0:
        """
        compute an epsilon graph from the similarities             
        for each node x_i, an epsilon graph has weights             
        w_ij = d(x_i,x_j) when w_ij >= eps, and 0 otherwise          
        """
        for i in range(n):
            for j in range(i):
                sim_ij = similarities[i, j]
                if sim_ij >= eps:
                    W[i, j] = sim_ij
                    W[j, i] = sim_ij

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
        similarities -= np.diag(np.diag(similarities))
        for i in range(n):
            neighbours = np.argsort(similarities[i, :])
            for NN in neighbours[-k:][::-1]:
                W[i, NN] = similarities[i, NN]

        W = np.maximum(W, W.T)  # to make W symmetric

    return W


def plot_similarity_graph(X, Y, var=1.0, eps=0.0, k=5):
    """
    Function to plot the similarity graph, given data and parameters.

    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n, ) vector with cluster assignments
    :param var:  the sigma value for the exponential function, already squared
    :param eps:  threshold eps for epsilon graphs
    :param k:    the number of neighbours k for k-nn
    :return:
    """
    # use the build_similarity_graph function to build the graph W
    # W: (n x n) dimensional matrix representing the adjacency matrix of the graph
    W = build_similarity_graph(X, var, eps, k)

    # Use auxiliary function to plot
    plot_graph_matrix(X, Y, W)


def how_to_choose_epsilon():
    """
    TO BE COMPLETED.

    Consider the distance matrix with entries dist(x_i, x_j) (the euclidean distance between x_i and x_j)
    representing a fully connected graph.
    One way to choose the parameter epsilon to build a graph is to choose the maximum value of dist(x_i, x_j) where
    (i,j) is an edge that is present in the minimal spanning tree of the fully connected graph. Then, the threshold
    epsilon can be chosen as exp(-dist(x_i, x_j)**2.0/(2*sigma^2)).
    """
    # the number of samples to generate
    num_samples = 100

    # the option necessary for worst_case_blob, try different values
    gen_pam = 2.0  # to understand the meaning of the parameter, read worst_case_blob in generate_data.py

    # get blob data
    X, Y = worst_case_blob(num_samples, gen_pam)

    # get two moons data
    # X, Y = two_moons(num_samples)
    n = X.shape[0]

    """
     use the distance function and the min_span_tree function to build the minimal spanning tree min_tree                   
     - var: the exponential_euclidean's sigma2 parameter          
     - dists: (n x n) matrix with euclidean distance between all possible couples of points                   
     - min_tree: (n x n) indicator matrix for the edges in the minimal spanning tree                           
    """
    var = 1.0
    dists = pairwise_distances(X).reshape((n, n))  # dists[i, j] = euclidean distance between x_i and x_j
    min_tree = min_span_tree(dists)

    """
    set threshold epsilon to the max weight in min_tree 
    """
    distance_threshold = np.max(dists[min_tree])
    eps = np.exp(- distance_threshold**2 / (2*var))

    """
    use the build_similarity_graph function to build the graph W  
     W: (n x n) dimensional matrix representing                    
        the adjacency matrix of the graph
       use plot_graph_matrix to plot the graph                    
    """
    W = build_similarity_graph(X, var=var, eps=eps, k=0)
    plot_graph_matrix(X, Y, W)


if __name__ == '__main__':
    n = 300
    blobs_data, blobs_clusters = blobs(n)
    moons_data, moons_clusters = two_moons(n)
    point_circle_data, point_circle_clusters = point_and_circle(n)
    worst_blobs_data, worst_blobs_clusters = worst_case_blob(n, 1.0)

    var = 1

    X, Y = moons_data, moons_clusters
    n_samples = X.shape[0]
    dists = pairwise_distances(X).reshape((n_samples, n_samples))
    min_tree = min_span_tree(dists)
    eps = np.exp(- np.max(dists[min_tree])**2 / (2*var))
    W_eps = build_similarity_graph(X, var=var, eps=0.6)
    W_knn = build_similarity_graph(X, k=15)

    plot_graph_matrix(X, Y, W_eps)
    plot_graph_matrix(X, Y, W_knn)