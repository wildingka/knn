import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    m = X.shape[0]
    n = Y.shape[0]
    euclid_dist = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            rowx = X[i,:]
            rowy = Y[j,:]
            euclid_dist[i,j] = np.linalg.norm(rowx-rowy, ord = None)
    return euclid_dist


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    m = X.shape[0]
    n = Y.shape[0]
    manhat_dist = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            rowx = X[i,:]
            rowy = Y[j,:]
            manhat_dist[i,j] = np.linalg.norm(rowx-rowy, ord = 1)
    return manhat_dist
   
    

def cos_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    m = X.shape[0]
    n = Y.shape[0]
    cos_dist = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            rowx = X[i,:]
            rowy = Y[j,:]
            numer = np.dot(np.transpose(rowx), rowy)
            denom = (np.linalg.norm(rowx, ord = 2) * np.linalg.norm(rowy, ord = 2))
            cos_dist[i,j] = 1 - float(numer/denom)
    return cos_dist

