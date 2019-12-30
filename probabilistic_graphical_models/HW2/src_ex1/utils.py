import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy.linalg import eigh


def plot_data_target_labels(X, labels, ax):
    """
    Plot data points X with one color for each label.
    """
    
    K_target = len(np.unique(labels)) # Number of target labels
    
    # define the colormap for the target labels
    cmap_target = plt.cm.cool
    # extract all colors from the .jet map
    colors_target = [cmap_target(i) for i in range(cmap_target.N)]
    N = len(colors_target)
    ndata = X.shape[0]
    unique_colors_target = [colors_target[int(i*N/K_target)] for i in range (K_target)]
    
    for idx_class, l in enumerate(np.unique(labels)):
        X_class_l = X[labels == l]
        sc_ = ax.scatter(*X_class_l.T, c = [unique_colors_target[l]], label="Actual label %s" % l, edgecolors = 'k', lw = .8)  
    
                             
                    
        
def plot_data_km_labels(X, labelsKM, ax):
    """
    Plot data points X with model labels.
    """
    
    K_km = len(np.unique(labelsKM)) # Number of K Means labels

    # define the colormap for the K Means labels
    cmap_km = plt.cm.autumn
    # extract all colors from the .jet map
    colors_km = [cmap_km(i) for i in range(cmap_km.N)]
    N = len(colors_km)
    ndata = X.shape[0]
    unique_colors_km = [colors_km[int(i*N/K_km)] for i in range (K_km)]
    
    for idx_class, l in enumerate(np.unique(labelsKM)):
        X_class_l = X[labelsKM == l]
        sc_ = ax.scatter(*X_class_l.T, c = 'w', edgecolors = [unique_colors_km[l]], lw = 1, alpha = 0.8, s = 90)
        # label = "KM label %s" % l, 
        
    return unique_colors_km
        


def plot_ellipse(mean, cov, ax, n_std=1., alpha=.2, facecolor=None):
    """
    Plot the confidence ellipsoid for a certain class.
    """

    # Get eigendecomposition of the ellipse
    lbda, v = eigh(cov)
    
    stdevs = n_std * (lbda ** .5)
    angle = np.degrees(np.arctan2(*v[:, 0][::-1]))
    
    ell = Ellipse(xy = mean, width = 2*stdevs[0], height = 2*stdevs[1], angle = angle, edgecolor = 'k', facecolor = facecolor, linewidth = 1., alpha = alpha, zorder = 1)
    ax.add_patch(ell)