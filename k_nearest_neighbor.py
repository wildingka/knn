import numpy as np 
from .distances import euclidean_distances, manhattan_distances, cosine_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.targets = None
        

    def fit(self, features, targets):
        """        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.features = features
        self.targets = targets
        

    def predict(self, features, ignore_first=False):
        """
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        n_samples = features.shape[0]
        n_dimensions = self.targets.shape[1] 
        labels = np.empty((n_samples, n_dimensions))
        for x in range(n_samples):
            example = np.array(features[x,:], ndmin= 2)   
            if (self.distance_measure == "euclidean"):
                D =  euclidean_distances(example, self.features)
            elif (self.distance_measure == "manhattan"):
                D =  manhattan_distances(example, self.features)
            elif (self.distance_measure == "cosine"):
                D = cosine_distances(example, self.features)
            D = np.argsort(D) #1 row by n_samples cols
            if (ignore_first == True):
                knn = D[0:,1:self.n_neighbors+1]
            else:
                knn = D[0:,0:self.n_neighbors]
            for y in range(n_dimensions):
                knntargets = self.targets[knn[0,:],y]
                if (self.aggregator == 'mode'):
                    aggregator = mymode
                if (self.aggregator == 'mean'):
                    aggregator = np.mean
                if (self.aggregator == 'median'):
                    aggregator = np.median
                prediction_result = aggregator(knntargets)
                labels[x,y] = prediction_result
        return labels


def mymode(targets):
    values, counts = np.unique(targets, return_counts=True)
    index = np.argmax(counts)
    mode = values[index]
    return mode


