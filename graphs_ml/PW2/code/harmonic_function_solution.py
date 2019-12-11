import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
from scipy.io import loadmat
import os

from helper import build_similarity_graph, build_laplacian, plot_classification, label_noise, \
                    plot_classification_comparison, plot_clusters, plot_graph_matrix, indices_to_one_hot

np.random.seed(50)


def build_laplacian_regularized(X, laplacian_regularization, var=1.0, eps=0.0, k=0, laplacian_normalization=""):
    """
    Function to construct a regularized Laplacian from data.

    :param X: (n x m) matrix of m-dimensional samples
    :param laplacian_regularization: regularization to add to the Laplacian (parameter gamma)
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: Q (n x n ) matrix, the regularized Laplacian
    """
    # build the similarity graph W
    W = build_similarity_graph(X, var, eps, k)

    """
    Build the Laplacian L and the regularized Laplacian Q.
    Both are (n x n) matrices.
    """
    L = build_laplacian(W, laplacian_normalization)

    # compute Q
    Q = L + laplacian_regularization*np.eye(W.shape[0])

    return Q


def mask_labels(Y, l):
    """
    Function to select a subset of labels and mask the rest.

    :param Y:  (n x 1) label vector, where entries Y_i take a value in [1, ..., C] , where C is the number of classes
    :param l:  number of unmasked (revealed) labels to include in the output
    :return:  Y_masked:
               (n x 1) masked label vector, where entries Y_i take a value in [1, ..., C]
               if the node is labeled, or 0 if the node is unlabeled (masked)
    """
    num_samples = np.size(Y, 0)

    """
     randomly sample l nodes to remain labeled, mask the others   
    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = indices[:l]

    Y_masked = np.zeros(num_samples)
    Y_masked[indices] = Y[indices]

    return Y_masked


def hard_hfs(X, Y, laplacian_regularization, var=1, eps=0, k=0, laplacian_normalization=""):
    """
    TO BE COMPLETED

    Function to perform hard (constrained) HFS.

    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [0, 1, ... , num_classes] (0 is unlabeled)
    :param laplacian_regularization: regularization to add to the Laplacian
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: labels, class assignments for each of the n nodes
    """

    Y = Y.astype(int)
    num_samples = np.size(X, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    """
    Build the vectors:
    l_idx = (l x 1) vector with indices of labeled nodes
    u_idx = (u x 1) vector with indices of unlabeled nodes
    """
    l_idx = np.where(Y != 0)[0]
    u_idx = np.where(Y == 0)[0]

    """
    Compute the hfs solution, remember that you can use the functions build_laplacian_regularized and 
    build_similarity_graph    
    
    f_l = (l x num_classes) hfs solution for labeled data. It is the one-hot encoding of Y for labeled nodes.   
    
    example:         
        if Cl=[0,3,5] and Y=[0,0,0,3,0,0,0,5,5], then f_l is a 3x2  binary matrix where the first column codes 
        the class '3'  and the second the class '5'.    
    
    In case of 2 classes, you can also use +-1 labels      
        
    f_u = array (u x num_classes) hfs solution for unlabeled data
    
    f = array of shape(num_samples, num_classes)
    """

    L = build_laplacian_regularized(X, laplacian_regularization, var, eps, k, laplacian_normalization)
    # Luu = L[u_idx, u_idx] does not work...
    Luu = L[u_idx, :]
    Luu = Luu[:, u_idx]
    # Lul = L[u_idx, l_idx] does not work...
    Lul = L[u_idx, :]
    Lul = Lul[:, l_idx]

    f_l = indices_to_one_hot(Y[l_idx] - 1, num_classes)
    f_u = - np.linalg.pinv(Luu) @ Lul @ f_l
    f = np.zeros((num_samples, num_classes))
    f[l_idx, :] = f_l
    f[u_idx, :] = f_u

    """
    compute the labels assignment from the hfs solution   
    labels: (n x 1) class assignments [1,2,...,num_classes]    
    """
    labels = np.argmax(f, axis=1) + 1

    return labels


def two_moons_hfs(small_dataset=True, method='hard', plot=True):
    """
    TO BE COMPLETED.

    HFS for two_moons data.
    """

    """
    Load the data. At home, try to use the larger dataset (question 1.2).    
    """
    # load the data
    if small_dataset:  # questions 1.1 and 1.3
        in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    else:  # question 1.2
        in_data = loadmat(os.path.join('data', 'data_2moons_hfs_large.mat'))

    X = in_data['X']
    Y = in_data['Y'].squeeze()

    # automatically infer number of labels from samples
    num_samples = np.size(Y, 0)
    num_classes = len(np.unique(Y))

    """
    Choose the experiment parameters
    """
    var = 1
    eps = 0
    if small_dataset:  # questions 1.1 and 1.3 parameter
        k = 12
    else:  # question 1.2 parameter
        k = 40
    laplacian_regularization = 0
    laplacian_normalization = "rw"
    c_l = .9
    c_u = .1

    # number of labeled (unmasked) nodes provided to the hfs algorithm
    l = 4

    # mask labels
    Y_masked = mask_labels(Y, l)

    """
    compute hfs solution using either soft_hfs or hard_hfs
    """
    if method == 'hard':
        labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    else: # if method == 'soft'
        labels = soft_hfs(X, Y_masked, c_l , c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    """
    Visualize results
    """
    if plot:
        if method == 'hard':
            method_name = 'Hard HFS'
        else:
            method_name = 'Soft HFS'

        plot_classification(X, Y, labels,  var=var, eps=0, k=k, method=method_name)

    accuracy = np.mean(labels == np.squeeze(Y))
    return accuracy

    
def soft_hfs(X, Y, c_l, c_u, laplacian_regularization, var=1, eps=0, k=0, laplacian_normalization=""):
    """
    TO BE COMPLETED.

    Function to perform soft (unconstrained) HFS


    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
    :param c_l: coefficients for C matrix
    :param c_u: coefficients for C matrix
    :param laplacian_regularization:
    :param var:
    :param eps:
    :param k:
    :param laplacian_normalization:
    :return: labels, class assignments for each of the n nodes
    """

    Y = Y.astype(int)
    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    """
    Compute the target y for the linear system  
    y = (n x num_classes) target vector 
    l_idx = (l x num_classes) vector with indices of labeled nodes    
    u_idx = (u x num_classes) vector with indices of unlabeled nodes 
    """

    l_idx = np.where(Y != 0)[0]
    u_idx = np.where(Y == 0)[0]

    y = np.zeros((num_samples, num_classes))
    y[l_idx] = indices_to_one_hot(Y[l_idx] - 1, num_classes)

    """
    compute the hfs solution, remember that you can use build_laplacian_regularized and build_similarity_graph
    f = (n x num_classes) hfs solution 
    C = (n x n) diagonal matrix with c_l for labeled samples and c_u otherwise    
    """

    Q = build_laplacian_regularized(X, laplacian_regularization, var, eps, k, laplacian_normalization)
    C = np.zeros(num_samples)
    C[l_idx] = c_l
    C[u_idx] = c_u
    C = np.diag(C)

    f = np.linalg.solve(np.linalg.pinv(C) @ Q + np.eye(num_samples), y)

    """
    compute the labels assignment from the hfs solution 
    labels: (n x 1) class assignments [1, ... ,num_classes]  
    """
    labels = np.argmax(f, axis=1) + 1

    return labels


def hard_vs_soft_hfs(plot=True):
    """
    TO BE COMPLETED.

    Function to compare hard and soft HFS.
    """
    # load the data
    in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    X = in_data['X']
    Y = in_data['Y'].squeeze()

    # automatically infer number of labels from samples
    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1
    
    # randomly sample 20 labels
    l = 20
    # mask labels
    Y_masked = mask_labels(Y, l)

    # Create some noisy labels
    Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], 4)

    """
    choose parameters
    """
    var = 1
    eps = 0
    laplacian_normalization = "rw"

    k = 15
    laplacian_regularization = .1
    c_l = .95
    c_u = .05

    """
    Compute hfs solution using soft_hfs() and hard_hfs().
    Remember to use Y_masked (the vector with some labels hidden as input and NOT Y (the vector with all labels 
    revealed)
    """
    hard_labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    soft_labels = soft_hfs(X, Y_masked, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    if plot:
        plot_classification_comparison(X, Y, hard_labels, soft_labels, var=var, eps=eps, k=k)

    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy


if __name__ == '__main__':

    # question 1.1 #
    print('----- Question 1.1 -----')
    np.random.seed(50)
    method = 'hard'
    acc = two_moons_hfs(small_dataset=True, method=method, plot=True)
    print('The accuracy on the small dataset with {} HFS is: {}\n'.format(method, acc))

    # question 1.2 #
    print('----- Question 1.2 -----')
    np.random.seed(33)  # to show that it can fail
    method = 'hard'
    acc = two_moons_hfs(small_dataset=False, method=method, plot=True)
    print('The accuracy on the large dataset with {} HFS is: {}\n'.format(method, acc))

    # question 1.3 #
    print('----- Question 1.3 -----')
    np.random.seed(50) # it seems it is not enough to get the same results at the end

    # test soft HFS
    method = 'soft'
    acc = two_moons_hfs(small_dataset=True, method=method, plot=True)
    print('The accuracy on the small dataset with {} HFS is: {}\n'.format(method, acc))

    # compare hard and soft HFS
    nb_measures = 10
    hard_vs_soft_acc = np.zeros((nb_measures, 2))

    hard_vs_soft_hfs(plot=True)  # to have one plot
    for k in range(nb_measures):
        hard_vs_soft_acc[k, :] = hard_vs_soft_hfs(plot=False)

    means = hard_vs_soft_acc.mean(axis=0)
    stds = hard_vs_soft_acc.std(axis=0)
    print('On an average of {} measures, the accuracies of hard HFS and soft HFS are:\n'.format(nb_measures) +
          '- Hard HFS: {} (std: {})\n'.format(means[0], stds[0]) +
          '- Soft HFS: {} (std: {})'.format(means[1], stds[1]))
