import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def sigmoid(x):
    """
    Sigmoid function, defined by sigmoid(x) = 1 / (1 + exp^{-x}).

    Parameters
    ----------
    x: ndarray

    Returns
    -------
    sigmoid_x: ndarray
    """
    return 1 / (1 + np.exp(-x))


def read_data(idx_dataset, train=True, test=True):
    """
    Read a dataset provided in the data folder.

    Parameters
    ----------
    idx_dataset: char
                  The dataset to open, i.e. 'A', 'B' or 'C'.

    train: bool
            Boolean to determine whether or not to open the train set (set to True if yes).

    test: bool
           Boolean to determine whether or not to open the test set (set to True if yes).

    Returns
    -------
    X_train: ndarray, shape (n_train, n_features) or None (if train=False)
              Train set.
              n_train == number of points in the train set.
              n_features == dimension of the points (e.g. each sample is in R^2).

    y_train: ndarray, shape (n_train,) or None (if train=False)
              Labels of the train set's examples.
              n_train == number of points in the train set.

    X_test: ndarray, shape (n_test, n_features) or None (if test=False)
              Test set.
              n_test == number of points in the test set.
              n_features == dimension of the points (e.g. each sample is in R^2).

    y_test: ndarray, shape (n_test,) or None (if test=False)
              Labels of the test set's examples.
              n_test == number of points in the test set.
    """

    if not train and not test:
        raise Exception('At least a dataset has to be opened.')

    if train:
        train_data = pd.read_csv(os.path.join('../', 'data', 'train'+idx_dataset), sep=" ", header=None)
        train_data = np.array(train_data)
        X_train = train_data[:, :2]
        y_train = train_data[:, 2]

    if test:
        test_data = pd.read_csv(os.path.join('../', 'data', 'test' + idx_dataset), sep=" ", header=None)
        test_data = np.array(test_data)
        X_test = test_data[:, :2]
        y_test = test_data[:, 2]

    if train and test:
        return X_train, y_train, X_test, y_test
    elif train and not test:
        return X_train, y_train, None, None
    elif not train and test:
        return None, None, X_test, y_test


def plot_results(clf, idx_dataset, X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Plot the results obtained with a model.

    Parameters
    ----------
    X_train: See read_data's documentation.

    y_train: See read_data's documentation.

    X_test: See read_data's documentation.

    y_test: See read_data's documentation.

    clf: LDA, LogisticRegression, LinearRegression or QDA
          Classifier to use for the results.
    """

    if X_train is None and X_test is None:
        raise Exception('You must provide at least a dataset.')

    # colors for the datasets
    train_colormap = ListedColormap(np.array(plt.cm.Paired.colors)[[0, 4]])
    test_colormap = ListedColormap(np.array(plt.cm.Paired.colors)[[1, 5]])

    plt.figure()

    # plot the training dataset
    if X_train is not None:
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=train_colormap)
        plt.scatter([], [], c=train_colormap.colors[0], label='Class 0 (train)')
        plt.scatter([], [], c=train_colormap.colors[1], label='Class 1 (train)')

    # plot the testing dataset
    if X_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, linewidth=1, edgecolors='k', cmap=test_colormap)
        plt.scatter([], [], c=test_colormap.colors[0], linewidth=1, edgecolors='k', label='Class 0 (test)')
        plt.scatter([], [], c=test_colormap.colors[1], linewidth=1, edgecolors='k', label='Class 1 (test)')

    if clf is not None:
        # retrieve the type of the classifier for the plot's title (e.g. LDA)
        type_clf = str(type(clf))
        type_clf = type_clf.replace('\'>', '')
        type_clf = type_clf[type_clf.rfind('.') + 1:]

        # plot the classifier's decision boundary
        xlim = plt.xlim()
        ylim = plt.ylim()

        xx = np.linspace(xlim[0], xlim[1], 50)
        yy = np.linspace(ylim[0], ylim[1], 50)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        plt.contour(XX, YY, Z, colors='k', levels=0.5, linestyles='-')
        if X_test is None:
            plt.title('{}'.format(type_clf))
        else:
            plt.title('{} - Accuracy on train (test) set {}: {} ({})'.format(type_clf,
                                                                             idx_dataset,
                                                                             round(clf.score(X_train, y_train), 4),
                                                                             round(clf.score(X_test, y_test), 4)))

    else:
        plt.title('Dataset {}'.format(idx_dataset))

    plt.legend(loc='best')
    plt.show()

    return
