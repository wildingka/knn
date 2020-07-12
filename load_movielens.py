import csv
import numpy as np
import os

def load_movielens_data(data_folder_path):
    """
    u.data -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	          user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC

    Args:
        data_folder_path {str}: Path to MovieLens dataset (given at data/ml-100).
    Returns:
        data {np.ndarray}: Numpy array of size 943x1682, with each item in the array
            containing the rating user i had for item j. If user i did not rate item j,
            the element (i, j) should be 0.
    """
    # This is the path to the file you need to load.
    data_file = os.path.join(data_folder_path, 'u.data')
    raw = np.genfromtxt(data_file, names= ('user_id','item_id','rating','timestamp') , dtype= int)
    people = np.unique(raw['user_id'])
    movies = np.unique(raw['item_id'])
    test = np.zeros((len(people),len(movies)))
    for person in people:
        for movie in movies:
            index = np.argwhere((raw['user_id']==person) & (raw['item_id'] == movie))
            if (index.size != 0):
                test[person-1,movie-1] =  raw['rating'][index]
    return test
    
