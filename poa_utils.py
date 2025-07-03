"""

This Python file provides necessary functions to create the figures in the
paper "Observable Covariance and Principal Observable Analysis", by 
Ece Karacam, Washington Mio and Osman Berat Okutan. 
 
Code by Ece Karacam and Osman Berat Okutan

"""

"""

Main functions to calculate principal observables and basis functions for observable domains.

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cvx
import dccp
from sklearn.metrics import pairwise_distances
from scipy.stats import uniform, norm
from time import time
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import scipy.io

def poa_network(G,k):
    """
    Args:
        G: networkx graph with N nodes. 
           Each node is assumed to have 1/N weight if not specified. 
           Each edge is assumed to have weight 1 if not specified, representing its length.
        k: Number of principal observables to construct.

    Returns:
        (k,N) numpy array, where the k_th row is the k_th principal observable.
    """
    N = len(G)  
    D = {}  
    weights = np.zeros((N,1))  
    
    for (node, data), i in zip(G.nodes(data=True), range(N)):
        D[node] = i
        weights[i] = data.get('weight', 1/N) 
    
    weights = weights / np.sum(weights)

    # Form the matrices for checking 1-Lipschitzness
    E = len(G.edges())  # Number of edges
    L = np.zeros((2 * E, N))
    d = np.zeros((2 * E, 1)) 
    for row, (v, w, r) in zip(range(E), G.edges(data=True)):
        i, j = D[v], D[w]
        L[2 * row][i] = 1
        L[2 * row][j] = -1
        L[2 * row + 1][i] = -1
        L[2 * row + 1][j] = 1
        d[2 * row, 0] = r.get('weight',1)
        d[2 * row + 1, 0] = r.get('weight', 1)
    
    # Create principal observables
    X = np.zeros((k, N))
    for i in range(k):
        f = cvx.Variable((N, 1)) # Variable to be optimized
        if i > 0:
            c = [(L @ f) <= d, cvx.sum(cvx.multiply(f, weights)) == 0, # Weighted mean POs must be centered
                 (X[:i, :] @ cvx.multiply(f, weights))  == 0]  # POs must be orthogonal
        else:
            c = [(L @ f) <= d, cvx.sum(cvx.multiply(f, weights)) == 0] 
     
        prob = cvx.Problem(cvx.Maximize(cvx.norm((cvx.multiply(f, weights**0.5)), 2)), c) # maximize the variance
        prob.solve(method='dccp', solver = 'ECOS')
        X[i] = f.value.transpose()
    return X


def poa_metric(X,k):
    """
    Args:
        X: (N,N) numpy array, a distance matrix.
        k: Number of principal observablse to constructed.

    Returns:
        (k,N) numpy array, where the k_th row is the k_th principal observable.
    """
    N=X.shape[0] 
        
    #Form the matrices for checking 1-Lipschitzness
    E=N*(N-1) #Number of conditions to check, 2 conditions per pair of distinct points
    L=np.zeros((2*E,N))
    d=np.zeros((2*E,1))
    row=-1
    for i in range(N-1):
        for j in range(i+1,N):
            row+=1
            L[row, i]=1
            L[row, j]=-1
            d[row,0]=X[i,j]
            row+=1
            L[row, i]=-1
            L[row, j]=1
            d[row]=X[i,j]
        
    #Create principle observables
    Xp=np.zeros((k,N))
    for i in range(k):
        f = cvx.Variable((N,1)) #Variable to be optimized
        c = [(L @ f) <= d, cvx.sum(f/N) == 0, (Xp[0:i]@f) == 0] if i>0 else [(L @ f) <= d, cvx.sum(f/N) == 0]
        prob = cvx.Problem(cvx.Maximize(cvx.norm(f,2)), c)
        prob.solve(solver=cvx.ECOS, method='dccp')
        Xp[i]=f.value.transpose()    
    return Xp


def po_basis(G, k): 
    """
    Args:
        G: networkx graph with N nodes.
           Each edge is assumed to have weight 1 if not specified, representing its length.
        k: Number of principal observables to construct.

    Returns:
        (k,N) numpy array, where the k_th row is the k_th basis function i.e. normalized 
        principal observables using L2 norm with respect to measure \mu.
    """
    N = len(G)  
    D = {} 
    weights = np.zeros((N,1))  
    
    # Retrieve node weights and construct the index map
    for (node, data), i in zip(G.nodes(data=True), range(N)):
        D[node] = i
        weights[i] = data.get('weight', 1/N)  # get node weights
    
    # Normalize node weights so they sum to 1 (if necessary)
    weights = weights / np.sum(weights)

    # Form the matrices for checking 1-Lipschitzness
    E = len(G.edges())  # number of edges
    L = np.zeros((2 * E, N))
    d = np.zeros((2 * E, 1))
    
    for row, (v, w, r) in zip(range(E), G.edges(data=True)):
        i, j = D[v], D[w]
        L[2 * row][i] = 1
        L[2 * row][j] = -1
        L[2 * row + 1][i] = -1
        L[2 * row + 1][j] = 1
        d[2 * row, 0] = r.get('weight',1)
        d[2 * row + 1, 0] = r.get('weight', 1)
    
    # Create principal observables
    X = np.zeros((k, N))
    for i in range(k):
        f = cvx.Variable((N, 1))  # variable to be optimized        
        # Define the constraints with weighted orthogonality and centering 
        if i > 0:
            c = [(L @ f) <= d, cvx.sum(cvx.multiply(f, weights)) == 0,   ### weighted mean must be zero 
                 (X[:i, :] @ cvx.multiply(f, weights))  == 0   ## Previous POs must be orthogonal
                ]
        else:
            c = [(L @ f) <= d, cvx.sum(cvx.multiply(f, weights)) == 0,  ### when only 1 PO is constructed 
                ]
     
        # Define the problem to maximize the weighted norm
        prob = cvx.Problem(cvx.Maximize(cvx.norm((cvx.multiply(f, weights**0.5)), 2)), c) ## finds the squared norm \sum f^2 \mu
        
        # Solve the optimization problem
        prob.solve(method='dccp', solver = 'ECOS')

        #NORMALIZATION STEP FOR GETTING ORTHONORMAL BASIS
        fval = f.value.flatten()
        mu = weights.flatten()
        current_norm = np.sqrt(np.sum(mu*fval**2))

        fval_normalised = fval/current_norm

        X[i] = fval_normalised.transpose()

    return X


"""
Helper functions
"""

def getImage(img, zoom=1):
    """
    Function to display images in the plot
    """
    return OffsetImage(img, zoom=zoom)


def metric_distortion(G, E, POA=False):  
    """
    Args:
        G: networkx graph with N nodes.
        E: Embedding of G.
           - If POA == True: E should be a (k, N) numpy array, where k is the number of principal observables and N is the number of nodes (each column is a node's embedding).
           - If POA == False: E should be a (N, d) numpy array, where N is the number of nodes and d is the embedding dimension (each row is a node's embedding).
        POA: Boolean. If True, uses Chebyshev (L-infinity) distance for pairwise distances in the embedded space (POA embedding).
             If False, uses Euclidean distance for pairwise distances in the embedded space (e.g., MDS embedding).

    Returns:
        numpy.ndarray: The distortion values (absolute differences between graph and embedding distances) for all unique node pairs.
    """
    D_G = nx.floyd_warshall_numpy(G)  # D_G is an (N, N) matrix where D_G[i, j] is the shortest path from node i to j
    triu_idx = np.triu_indices(D_G.shape[0], k=1) # Get the indices for the upper triangle of the matrix, excluding the diagonal (k=1)
    upper_triangle_graph = D_G[triu_idx] # All unique pairwise shortest path distances (i < j)
    if POA:
        # POA embedding
        if E.shape[1] != D_G.shape[0]:
            raise ValueError(f"Expected E to have shape (k, N) where N={D_G.shape[0]}, got {E.shape}")
        D_E = pairwise_distances(E.T, metric='chebyshev')
        upper_triangle_embedded = D_E[triu_idx] 
        distortion = np.abs(upper_triangle_graph - upper_triangle_embedded)
        return  distortion
    else:
        # MDS embedding
        D_E = pairwise_distances(E, metric='euclidean')
        upper_triangle_embedded = D_E[triu_idx]
        distortion = np.abs(upper_triangle_graph - upper_triangle_embedded)
        return distortion 