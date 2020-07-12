import numpy as np
from .k_nearest_neighbor import KNearestNeighbor


def collaborative_filtering(input_array, n_neighbors,
                            distance_measure='euclidean', aggregator='mode'):
    """
       Arguments:
        input_array {np.ndarray} -- An input array of shape (n_samples, n_features).
            Any zeros will get imputed.
        n_neighbors {int} -- Number of neighbors to use for prediction.
        distance_measure {str} -- Which distance measure to use. Can be one of
            'euclidean', 'manhattan', or 'cosine'. This is the distance measure
            that will be used to compare features to produce labels.
        aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
            neighbors. Can be one of 'mode', 'mean', or 'median'.

    Returns:
        imputed_array {np.ndarray} -- An array of shape (n_samples, n_features) with imputed
            values for any zeros in the original input_array.
    """
    n_samples = input_array.shape[0]
    n_dimensions = input_array.shape[1] 
    KNN = KNearestNeighbor(n_neighbors=n_neighbors, distance_measure=distance_measure, aggregator=aggregator)
    KNN.fit(input_array, input_array)
    imputed_array = np.empty((n_samples,n_dimensions))
    for row in range(n_samples):
        example = input_array[[row],:]
        output = KNN.predict(example,ignore_first=True)
        imputed_array[row] = np.where(example == 0, output, example)
    return imputed_array

    
