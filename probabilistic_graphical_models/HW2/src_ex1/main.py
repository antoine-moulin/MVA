import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from itertools import combinations
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle

from diagGaussianEM import diagGaussianEM
from utils import plot_data_target_labels, plot_data_km_labels, plot_ellipse


## Data

dataCustom = True
dataCustomN = 2

if not(dataCustom):
    dataIris = load_iris()
    X = dataIris.data
    labelsTarget = dataIris.target
    res_path = "../report/iris"

if dataCustom:
    if (dataCustomN == 1) :
        x = np.append(np.random.randint(0,800,1000), np.random.randint(925,1425,500))
        y = np.random.rand(1500)/1000
        X = np.stack((x,y),axis = 0).T
        labelsTarget = np.append(np.array([0] * 1000), np.array([1] * 500))
    else:
        x = np.append(np.random.normal(0,0.9,500), np.random.normal(0,0.2,500))
        y = np.append(np.random.normal(0,0.4,500), np.random.normal(0,0.8,500))
        X = np.stack((x,y),axis = 0).T
        labelsTarget = np.append(np.array([0] * 500), np.array([1] * 500))
        
    res_path = "../report/custom"

os.makedirs(res_path, exist_ok=True)


## Class Prediction

K = 2 #Number of classes

# Gaussian Mixture (diagonal covariance matrices)
p, musDiagGM, sigmasDiagGM, labelsDiagGM = diagGaussianEM(X, K)

# Gaussian Mixture (full covariance matrices)
gm = GaussianMixture(n_components=K)
gm.fit(X)
labelsFullGM = gm.predict(X)
musFullGM = gm.means_
sigmasFullGM = gm.covariances_

# K Means
km = KMeans(n_clusters=K)
km.fit(X)
labelsKM = km.predict(X)
centroidsKM = km.cluster_centers_



if __name__ == '__main__':
    

    dim = X.shape[1]
    dim_pairs = list(combinations(range(dim), 2)) #every possible pair of dimensions
    
    nrows = len(dim_pairs)
    ncols = 3
    
    if dataCustom:
        figsize = (15, 5)
    else:
        figsize = (15,20)
    
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize)
    
    for j in range(nrows):
        dim_pair = list(list(combinations(range(dim), 2))[j])
        X_2d = X[:, dim_pair]
        
        for i in range(ncols):
            if (nrows == 1) :
                ax = axes[i]
            else:
                ax = axes[j,i]
            
            if (i == 0):
                #Gaussian mixture: diagonal covariance matrices
                unique_colors_DiagGM = plot_data_km_labels(X_2d, labelsDiagGM, ax)
                plot_data_target_labels(X_2d, labelsTarget, ax)
                for k in range (K) :
                    mus_dim_k = musDiagGM[k,:][dim_pair]
                    sigmas_dim_k = sigmasDiagGM[k,:,:][np.ix_(dim_pair, dim_pair)]
                    
                    sc_ = ax.scatter(*mus_dim_k, marker = "v", label="Class %d" % (k+1),edgecolor='w', s=100, c = [unique_colors_DiagGM[k]])
                    col = sc_.get_facecolor()[0]                  
                    plot_ellipse(mus_dim_k, sigmas_dim_k, ax, n_std = K, facecolor = col)

                ax.legend(facecolor='white')
                ax.set_title("Diagonal Gaussian mixture model (dimensions %s)" % list(dim_pair))
                
                
                
                
                
            if (i == 1):
                #Gaussian mixture: full covariance matrices
                unique_colors_FullGM = plot_data_km_labels(X_2d, labelsFullGM, ax)
                plot_data_target_labels(X_2d, labelsTarget, ax)
                for k in range (K) :
                    mus_dim_k = musFullGM[k,:][dim_pair]
                    sigmas_dim_k = sigmasFullGM[k,:,:][np.ix_(dim_pair, dim_pair)]
                    
                    sc_ = ax.scatter(*mus_dim_k, marker = "v", label="Class %d" % (k+1),edgecolor='w', s=100, c = [unique_colors_FullGM[k]])
                    col = sc_.get_facecolor()[0]                  
                    plot_ellipse(mus_dim_k, sigmas_dim_k, ax, n_std = K, facecolor = col)

                ax.legend(facecolor='white')
                ax.set_title("Full Gaussian mixture model (dimensions %s)" % list(dim_pair))
                
                
            if (i == 2):
                #K Means
                unique_colors_km = plot_data_km_labels(X_2d, labelsKM, ax)
                plot_data_target_labels(X_2d, labelsTarget, ax)
                for k in range (K) :
                    centroid_dim_k = centroidsKM[k,:][dim_pair]
                    radius = cdist(X_2d[labelsKM == k], [centroid_dim_k]).max()
                    sc_ = ax.scatter(*centroid_dim_k, marker = "v", label="Class %d" % (k+1),edgecolor='w', s=100, c = [unique_colors_km[k]])
                    col = sc_.get_facecolor()[0]                  
                    cir = Circle(xy = centroid_dim_k, radius = radius, zorder = 1, edgecolor = 'k', facecolor = col, linewidth = 1., alpha = .2)
                    ax.add_patch(cir)

                    
                ax.legend(facecolor='white')
                ax.set_title("K Means (dimensions %s)" % list(dim_pair))
    
    fig.suptitle("Comparison of the models for K = %s" %K)
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85, hspace=0.3)
    
    if dataCustom:
        fig.savefig(res_path + '/custom%d_GM_KM_%d.pdf' % (dataCustomN,K))
    else:
        fig.savefig(res_path + '/iris_GM_KM_%d.pdf' % K)

    plt.show()