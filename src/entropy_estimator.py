#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#

import ray
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import BallTree
import warnings

#-----------------------------------------------------------------------------------------------------------------------------------------#

def entropy(x, p, rho):
    return np.trapz(np.where(p(x) > 0, - p(x) * np.log(p(x)/rho(x)), 0), x)

#-----------------------------------------------------------------------------------------------------------------------------------------#

def classificability_estimator(x_data, y_data, distance_threshold=None, k_neighbors=None, return_probs=False, return_entropies=False, metric='minkowski', num_workers=4):

    @ray.remote
    def query_balltree(query_chunk, x, y, n_c):
        # Build a Ball Tree for efficient distance estimation.
        worker_tree = BallTree(x, metric=metric)
        if k_neighbors is not None:
            indices = worker_tree.query(query_chunk, k=k_neighbors, return_distance=False)
        else:
            indices = worker_tree.query_radius(query_chunk, r=distance_threshold, return_distance=False)
        # Estimate probability around each point.
        probs = np.zeros((query_chunk.shape[0], n_c))
        for it, idx in enumerate(indices):
            unique, counts = np.unique(y[idx], return_counts=True)
            probs[it, unique] = counts / np.sum(counts)
        return probs

    assert distance_threshold != k_neighbors, 'Either distance_threshold or k_neighbors must be defined.'
    assert distance_threshold == None or k_neighbors == None, 'Either distance_threshold or k_neighbors must be None.'

    num_classes = len(np.unique(y_data))
    # Reshape if array is 1D for the Ball Tree.
    # The first dim corresponds to sample index.
    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1,1)

    # Share data among workers.
    share_x = ray.put(x_data)
    share_y = ray.put(y_data)
    query_x_chunks = np.array_split(x_data, num_workers)
    #ray.init(ignore_reinit_error=True)
    results = ray.get([query_balltree.remote(x_chunk, share_x, share_y, num_classes) for x_chunk in query_x_chunks])
    #ray.shutdown()
    probs = np.vstack(results)
    # Compute entropy assuming constant entropy around each data point.
    entropy = np.where(probs > 0, -probs*np.log( probs / ( probs*np.exp(1-np.max(probs, axis=1).reshape(-1,1) ) ) ), 0)
    # Classificability is the estimated as 1 minus the mean entropy.
    classificability = 1 - np.mean(np.sum(entropy, axis=1))
    # Deliver output
    output = [classificability]
    output = output + [probs] if return_probs else output
    output = output + [entropy] if return_entropies else output
    return output[0] if len(output) == 1 else tuple(output)

#-----------------------------------------------------------------------------------------------------------------------------------------#

def mean_min_max_dist_estimation(x_data, k_low=2, k_high=10, metric='minkowski', num_workers=4):

    @ray.remote
    def query_balltree(query_chunk, x, k):
        # Build a Ball Tree for efficient distance estimation.
        worker_tree = BallTree(x, metric=metric)
        distances, _ = worker_tree.query(query_chunk, k=k, return_distance=True)
        return distances[:, 1:]

    if k_high is not None:
        assert k_low < k_high, 'k_high must be larger than k_low'

    #ray.init(ignore_reinit_error=True)

    # Reshape if array is 1D for the Ball Tree.
    # The first dim corresponds to sample index.
    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1,1)

    # Share data among workers.
    share_x = ray.put(x_data)
    query_x_chunks = np.array_split(x_data, num_workers)
    
    # k_low distances.
    results = ray.get([query_balltree.remote(x_chunk, share_x, k_low) for x_chunk in query_x_chunks])
    low_distances = np.vstack(results)
    # k_high distances.
    results = ray.get([query_balltree.remote(x_chunk, share_x, k_high) for x_chunk in query_x_chunks])
    high_distances = np.vstack(results)
    
    # Estimate mean distance.
    mean_low_x = np.mean(low_distances)  
    mean_high_x = np.mean(high_distances)  

    #ray.shutdown()

    return mean_low_x, mean_high_x

#-----------------------------------------------------------------------------------------------------------------------------------------#

def density_dist_space(x_data, points=25, d_low=0.001, d_high=0.05, metric='minkowski', num_workers=4):


    @ray.remote
    def query_balltree(query_chunk, x, k):
        # Build a Ball Tree for efficient distance estimation.
        worker_tree = BallTree(x, metric=metric)
        distances, _ = worker_tree.query(query_chunk, k=k, return_distance=True)
        return distances[:, 1:]

    assert d_low < d_high, 'd_high must be larger than d_low'
    # Reshape if array is 1D for the Ball Tree.
    # The first dim corresponds to sample index.
    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1,1)

    # Share data among workers.
    share_x = ray.put(x_data)
    query_x_chunks = np.array_split(x_data, num_workers)

    # Estimate mean distance of k_low nearest neighbors to each point.
    density_thresholds = np.linspace(d_low, d_high, points)
    mean_distances = []
    for i in range(points):
        k = round(density_thresholds[i] * x_data.shape[0])
        if k < 2:
            warnings.warn(f'Invalid low threshold, k set to 2 for density {density_thresholds[i]}.')
            k = 2
        results = ray.get([query_balltree.remote(x_chunk, share_x, k) for x_chunk in query_x_chunks])
        distances = np.vstack(results)
        mean_x = np.mean(distances)  
        mean_distances.append(mean_x)
    return density_thresholds, mean_distances

#-----------------------------------------------------------------------------------------------------------------------------------------#

def density_dist(x_data, d=0.01, min_k=None, metric='minkowski'):
    # Reshape if array is 1D for the Ball Tree.
    # The first dim corresponds to sample index.
    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1,1)
    # Build a Ball Tree for efficient distance estimation.
    tree = BallTree(x_data, metric=metric)
    # Estimate mean distance of round(d * len(x_data)) nearest neighbors to each point.
    k = max(round(d * x_data.shape[0]), min_k) if min_k is not None else round(d * x_data.shape[0])
    if k < 2:
        warnings.warn(f'Invalid low threshold and min_k is None. k set to 2 for density.')
        k = 2
    distances, _ = tree.query(x_data, k=k)
    mean_x = np.mean(distances[:, 1:])  
    return mean_x

#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#