"""
Code for generating data in R^2 for clustering algorithms.
"""

import numpy as np
import sklearn.datasets as skd


def worst_case_blob(num_samples, gen_pam):
    """
    Generates a single blob.

    :param num_samples: number of samples to create in the blob
    :param gen_pam: distance of the outlier from the blob
    :return: X,  (num_samples, 2) matrix of 2-dimensional samples
             Y,  (num_samples, ) vector of "true" cluster assignment
    """
    blob_var = 0.3
    # X: matrix of shape (num_samples, 2)
    # Y: vector of shape (num_samples, )
    X, Y = skd.make_blobs(n_samples=num_samples, n_features=2, centers=np.column_stack((0, 0)), cluster_std=blob_var)
    X[-1] = [np.max(X) + gen_pam, 0]
    return [X, Y]


def blobs(num_samples, n_blobs=2, blob_var=0.15, surplus=0):
    """
    Creates N gaussian blobs evenly spaced across a circle.
    :param num_samples: number of samples to create in the dataset
    :param n_blobs:      how many separate blobs to create
    :param blob_var:    gaussian variance of each blob
    :param surplus:     number of extra samples added to first blob to create unbalanced classes
    :return: X,  (num_samples, 2) matrix of 2-dimensional samples
             Y,  (num_samples, ) vector of "true" cluster assignment
    """
    # data array
    X = np.zeros((num_samples, 2))
    # array containing the indices of the true clusters
    Y = np.zeros(num_samples, dtype=np.int32)

    # generate data
    block_size = (num_samples-surplus)//n_blobs

    for ii in range(1, n_blobs+1):
        start_index = (ii - 1) * block_size
        end_index = ii * block_size
        if ii == n_blobs:
            end_index = num_samples
        Y[start_index:end_index] = ii - 1
        nn = end_index - start_index

        X[start_index:end_index, 0] = np.cos(2*np.pi*ii/n_blobs) + blob_var*np.random.randn(nn)
        X[start_index:end_index, 1] = np.sin(2*np.pi*ii/n_blobs) + blob_var*np.random.randn(nn)
    return X, Y


def two_moons(num_samples, moon_radius=2.0, moon_var=0.02):
    """
    Creates two intertwined moons

    :param num_samples: number of samples to create in the dataset
    :param moon_radius: radius of the moons
    :param moon_var:    variance of the moons
    :return: X,  (num_samples, 2) matrix of 2-dimensional samples
             Y,  (num_samples, ) vector of "true" cluster assignment
    """
    X = np.zeros((num_samples, 2))

    for i in range(int(num_samples / 2)):
        r = moon_radius + 4 * i / num_samples
        t = i * 3 / num_samples * np.pi
        X[i, 0] = r * np.cos(t)
        X[i, 1] = r * np.sin(t)
        X[i + int(num_samples / 2), 0] = r * np.cos(t + np.pi)
        X[i + int(num_samples / 2), 1] = r * np.sin(t + np.pi)

    X = X + np.sqrt(moon_var) * np.random.normal(size=(num_samples, 2))
    Y = np.ones(num_samples)
    Y[:int(num_samples / 2) + 1] = 0
    return [X, Y.astype(int)]


def point_and_circle(num_samples, radius=2.0, sigma=0.15):
    """
    Creates point and circle

    :param num_samples: number of samples to create in the dataset
    :param sigma:       variance
    :param radius:      radius of the circle
    :return: X,  (num_samples, 2) matrix of 2-dimensional samples
             Y,  (num_samples, ) vector of "true" cluster assignment [1:c]
    """
    # data array
    X = np.zeros((num_samples, 2))
    # array containing the indices of the true clusters
    Y = np.zeros(num_samples, dtype=np.int32)

    # generate data
    block_size = num_samples // 2
    for ii in range(1, 3):
        start_index = (ii - 1) * block_size
        end_index = ii * block_size
        if ii == 3:
            end_index = num_samples
        Y[start_index:end_index] = ii - 1
        nn = end_index - start_index
        if ii == 1:
            X[start_index:end_index, 0] = sigma*np.random.randn(nn)
            X[start_index:end_index, 1] = sigma*np.random.randn(nn)
        else:
            angle = 2*np.pi * np.random.uniform(size=nn) - np.pi
            X[start_index:end_index, 0] = radius*np.cos(angle) + sigma * np.random.randn(nn)
            X[start_index:end_index, 1] = radius*np.sin(angle) + sigma * np.random.randn(nn)
    return X, Y


# --------------------------------
# Visualizing the data
# --------------------------------

if __name__ == '__main__':
    from utils import plot_clusters

    blobs_data, blobs_clusters = blobs(600)
    moons_data, moons_clusters = two_moons(600)
    point_circle_data, point_circle_clusters = point_and_circle(600)
    worst_blobs_data, worst_blobs_clusters = worst_case_blob(600, 1.0)

    plot_clusters(blobs_data, blobs_clusters, 'blobs', show=False)
    plot_clusters(moons_data, moons_clusters, 'moons', show=False)
    plot_clusters(point_circle_data, point_circle_clusters, 'point and circle', show=False)
    plot_clusters(worst_blobs_data, worst_blobs_clusters, 'worst case blob', show=True)


