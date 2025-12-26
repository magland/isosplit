import os
from typing import Tuple, Union, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .density_estimation import DensityEstimationConfig, estimate_density_dip
from .isosplit_verbose import (
    VERBOSE,
    print_header,
    print_iteration_info,
    print_separation_info,
    print_decision_merge,
    print_decision_redistribute,
    plot_clusters,
    plot_decision,
    plot_1d_histogram,
)

VERBOSE = os.environ.get("ISOSPLIT_VERBOSE", "0") == "1"

class Timer:
    def __init__(self, name: str, profile_dict: dict):
        self.name = name
        self.profile_dict = profile_dict

    def __enter__(self):
        self.start_time = os.times()[4]  # User + System time

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = os.times()[4]
        elapsed = end_time - self.start_time
        if not self.name in self.profile_dict:
            self.profile_dict[self.name] = 0.0
        self.profile_dict[self.name] += elapsed

def print_timings(profile_dict: dict) -> None:
    if VERBOSE or True:
        print("Timing profile:")
        for key, value in profile_dict.items():
            print(f"  {key}: {value:.4f} seconds")
        print("")

def isosplit(
    data: np.ndarray,
    *,
    separation_threshold: float = 2,
    initial_k: int = 30,
    density_config: Optional[DensityEstimationConfig] = None,
    use_lda_for_merge_test: bool = True
) -> np.ndarray:
    """
    IsoSplit clustering algorithm.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n, d) where n is the number of samples
        and d is the number of dimensions.
    separation_threshold : float, default=2
        Threshold for determining whether clusters should be merged.
        Higher values result in more clusters.
    initial_k : int, default=30
        Number of initial clusters for k-means initialization.
    density_config : DensityEstimationConfig or None, default=None
        Configuration for 1D density estimation. If None, uses default GMM method.
    use_lda_for_merge_test : bool, default=True
        Whether to use LDA for finding the optimal projection direction
        during the merge test. If False, uses simple centroid-based direction.

    Returns
    -------
    np.ndarray
        Cluster labels of shape (n,) with cluster assignments.
        Labels are integers starting from 1.
        
    Examples
    --------
    >>> import numpy as np
    >>> from isosplit import isosplit, DensityEstimationConfig
    >>> X = np.random.randn(100, 2)
    >>> labels = isosplit(X)
    >>> 
    >>> # Use faster KDE method
    >>> config = DensityEstimationConfig(method='kde')
    >>> labels = isosplit(X, density_config=config)
    """
    if density_config is None:
        density_config = DensityEstimationConfig()
    
    profile_timing = {}
    n_samples, n_features = data.shape

    # Step 1: Initial k-means clustering with large k
    if initial_k == n_samples:
        labels = np.arange(1, n_samples + 1)  # Each point its own cluster
    else:
        if initial_k > n_samples:
            initial_k = n_samples  # Cannot have more clusters than samples
        with Timer('kmeans', profile_timing):
            # We use n_init=1 for speed
            kmeans = KMeans(n_clusters=initial_k, random_state=1, n_init=1)
            labels = kmeans.fit_predict(data) + 1  # Convert to 1-based indexing

    if VERBOSE:
        print_header(initial_k, separation_threshold, data.shape)
        plot_clusters(data, labels, "Initial K-means Clustering", 0, 0)
    
    num_clusters = len(np.unique(labels))
    active_cluster_ids = np.arange(1, num_clusters + 1)
    
    cluster_indices = {}
    for cluster_id in active_cluster_ids:
        cluster_indices[cluster_id] = np.where(labels == cluster_id)[0]

    # Step 2: Iterative merging
    for passnum in [1, 2]:
        with Timer('part2a', profile_timing):
            iteration = 0

            # Initialize centroids and distance matrix once at the start of each pass
            with Timer('initial_compute_centroids', profile_timing):
                centroids = _compute_centroids(data, labels, num_clusters)
            with Timer('initial_compute_distance_matrix', profile_timing):
                distances = _compute_distance_matrix(centroids)

        while True:
            with Timer('part2b', profile_timing):
                if len(active_cluster_ids) <= 1:
                    break

                # find mutually closest pairs
                with Timer('find_mutually_closest_pairs', profile_timing):
                    pairs = _find_mutually_closest_pairs(distances, active_cluster_ids)
                    if not pairs:
                        break

            cluster_ids_to_update = []
            cluster_ids_to_remove = []
            for pair in pairs:
                with Timer('part2c', profile_timing):
                    with Timer('misc', profile_timing):
                        cluster1_id, cluster2_id = pair
                        iteration += 1

                        # Get data points for each cluster
                        cluster1_data = data[cluster_indices[cluster1_id]]
                        cluster2_data = data[cluster_indices[cluster2_id]]

                    if VERBOSE:
                        print_iteration_info(
                            passnum,
                            iteration,
                            cluster1_id,
                            cluster2_id,
                            len(cluster1_data),
                            len(cluster2_data),
                        )

                    # Perform merge test with configured density estimation
                    centroid1 = centroids[cluster1_id - 1]
                    centroid2 = centroids[cluster2_id - 1]
                    separation_score, cutpoint, new_assignments, cluster1_1d, cluster2_1d = (
                        _merge_test(cluster1_data, cluster2_data, centroid1, centroid2, density_config, use_lda_for_merge_test, profile_timing)
                    )

                if VERBOSE:
                    print_separation_info(separation_score, separation_threshold)

                with Timer('part2d', profile_timing):
                    # Decision: merge or redistribute
                    if separation_score > separation_threshold:
                        # set distance between these to inf to avoid reprocessing
                        distances[cluster1_id - 1, cluster2_id - 1] = np.inf
                        distances[cluster2_id - 1, cluster1_id - 1] = np.inf
                        # Keep clusters separate but redistribute points based on assignments from 1D test
                        if VERBOSE:
                            old_labels = labels.copy()

                        # Apply new assignments directly from the 1D redistribution
                        # new_assignments is ordered as [cluster1_points, cluster2_points]
                        # Split the assignments back to match the original clusters
                        n1 = len(cluster1_data)
                        assignments_cluster1 = new_assignments[:n1]
                        assignments_cluster2 = new_assignments[n1:]

                        changed = np.any(assignments_cluster1 != 0) or np.any(assignments_cluster2 != 1)
                        if changed:
                            cluster_ids_to_update.append(cluster1_id)
                            cluster_ids_to_update.append(cluster2_id)

                            # Map 0/1 to cluster IDs
                            labels[cluster_indices[cluster1_id]] = np.where(
                                assignments_cluster1 == 0, cluster1_id, cluster2_id
                            )
                            labels[cluster_indices[cluster2_id]] = np.where(
                                assignments_cluster2 == 0, cluster1_id, cluster2_id
                            )
                            # Update cluster indices
                            cluster_indices[cluster1_id] = np.where(labels == cluster1_id)[0]
                            cluster_indices[cluster2_id] = np.where(labels == cluster2_id)[0]

                        if VERBOSE:
                            print_decision_redistribute(changed)
                            plot_1d_histogram(
                                cluster1_1d,
                                cluster2_1d,
                                cutpoint,
                                separation_score,
                                cluster1_id,
                                cluster2_id,
                                "Redistributed",
                                iteration,
                                passnum,
                            )
                            plot_decision(
                                data,
                                old_labels,
                                labels,
                                cluster1_id,
                                cluster2_id,
                                "Redistributed",
                                iteration,
                                passnum,
                            )
                    else:
                        # Merge clusters - assign all points to the smaller cluster ID
                        merge_to = min(cluster1_id, cluster2_id)
                        merge_from = max(cluster1_id, cluster2_id)
                        cluster_ids_to_update.append(merge_to)
                        cluster_ids_to_remove.append(merge_from)

                        if VERBOSE:
                            print_decision_merge(merge_to)
                            old_labels = labels.copy()

                        labels[cluster_indices[merge_from]] = merge_to
                        # Update cluster indices to union of both clusters
                        cluster_indices[merge_to] = np.union1d(cluster_indices[merge_to], cluster_indices[merge_from])
                        del cluster_indices[merge_from]

                        if VERBOSE:
                            plot_1d_histogram(
                                cluster1_1d,
                                cluster2_1d,
                                cutpoint,
                                separation_score,
                                cluster1_id,
                                cluster2_id,
                                "Merged",
                                iteration,
                                passnum,
                            )
                            plot_decision(
                                data,
                                old_labels,
                                labels,
                                cluster1_id,
                                cluster2_id,
                                "Merged",
                                iteration,
                                passnum,
                            )
            
            with Timer('part2e', profile_timing):
                # Update active cluster IDs to exclude removed clusters
                if cluster_ids_to_remove:
                    active_cluster_ids = np.setdiff1d(
                        active_cluster_ids, cluster_ids_to_remove
                    )

                if cluster_ids_to_update:
                    # update centroids for the cluster_ids_to_update
                    with Timer('update_centroids', profile_timing):
                        for cluster_id in set(cluster_ids_to_update):
                            mask = labels == cluster_id
                            cluster_data = data[mask]
                            centroids[cluster_id - 1] = np.mean(cluster_data, axis=0)
                    
                    # update distance matrix for the cluster_ids_to_update
                    with Timer('update_distance_matrix', profile_timing):
                        _update_distance_matrix(
                            distances, centroids, active_cluster_ids, cluster_ids_to_update
                        )

    with Timer('part3', profile_timing):
        # Step 3: Reassign labels to be consecutive with no gaps
        labels = _reassign_labels(labels)

    print_timings(profile_timing)

    return labels

def _update_distance_matrix(
    distances: np.ndarray,
    centroids: np.ndarray,
    active_cluster_ids: np.ndarray,
    cluster_ids_to_update: list[int]
) -> None:
    """
    Update only the rows/cols for changed clusters in the distance matrix.
    Uses SQUARED distances and vectorized operations for efficiency.
    Uses the invariant index = label - 1.
    Modifies the distances array in place.

    Parameters:
    -----------
    distances : np.ndarray
        SQUARED distance matrix to update
    centroids : np.ndarray
        Array of centroids where centroids[label-1] is the centroid for that label
    active_cluster_ids : np.ndarray
        Array of currently active cluster IDs
    cluster_ids_to_update : list[int]
        List of cluster IDs to update
    """
    for cluster_id in set(cluster_ids_to_update):
        sq_dists = np.sum(
            (centroids[active_cluster_ids - 1] - centroids[cluster_id - 1]) ** 2,
            axis=1
        )
        # preserve inf values for already compared pairs
        sq_dists[distances[cluster_id - 1, active_cluster_ids - 1] == np.inf] = np.inf
        distances[cluster_id - 1, active_cluster_ids - 1] = sq_dists
        distances[active_cluster_ids - 1, cluster_id - 1] = sq_dists
        distances[cluster_id - 1, cluster_id - 1] = np.inf  # Diagonal

def _compute_centroids(
    data: np.ndarray, labels: np.ndarray, num_clusters: int
) -> np.ndarray:
    """
    Compute centroids for each cluster.

    Parameters:
    -----------
    data : np.ndarray
        Input data array
    labels : np.ndarray
        Current cluster labels (1-based)
    num_clusters : int
        Number of clusters

    Returns:
    --------
    centroids : np.ndarray
        Array of shape (num_clusters, n_features) where centroids[label-1] is the centroid
        for cluster with that label
    """
    n_features = data.shape[1]
    centroids = np.zeros((num_clusters, n_features))

    for label in range(1, num_clusters + 1):
        mask = labels == label
        if np.any(mask):
            centroids[label - 1] = np.mean(data[mask], axis=0)

    return centroids

def _compute_distance_matrix(
    centroids: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise SQUARED distances between all centroids using vectorized operations.
    Uses the invariant that index = label - 1.
    
    NOTE: Returns squared distances to avoid expensive sqrt operations.

    Parameters:
    -----------
    centroids : np.ndarray
        Array of centroids where centroids[label-1] is the centroid for that label

    Returns:
    --------
    distances : np.ndarray
        SQUARED distance matrix of shape (max_label, max_label) where distances[label1-1, label2-1]
        is the SQUARED distance between clusters label1 and label2
    """
    max_label = centroids.shape[0]
    distances = np.zeros((max_label, max_label))

    # Compute pairwise squared distances using broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_norms = np.sum(centroids**2, axis=1, keepdims=True)
    sq_distances = (
        sq_norms + sq_norms.T - 2 * np.dot(centroids, centroids.T)
    )

    # Ensure non-negative (numerical precision issues)
    sq_distances = np.maximum(sq_distances, 0)

    # Place computed SQUARED distances in the full matrix (no sqrt)
    distances[:, :] = sq_distances

    # Set diagonal to inf to avoid self-comparison
    for i in range(max_label):
        distances[i, i] = np.inf

    return distances

def _find_mutually_closest_pairs(
    distances: np.ndarray, active_cluster_ids: np.ndarray
) -> list[Tuple[int, int]]:
    distance_submatrix = distances[np.ix_(active_cluster_ids - 1, active_cluster_ids - 1)]
    n = len(active_cluster_ids)
    argmins1 = np.argmin(distance_submatrix, axis=1)
    argmins2 = np.argmin(distance_submatrix, axis=0)
    pairs = []
    for i in range(n):
        j = argmins1[i]
        if argmins2[j] == i and i < j:
            label_i = active_cluster_ids[i]
            label_j = active_cluster_ids[j]
            pairs.append((label_i, label_j))
    return pairs

def _merge_test(
    cluster1_data: np.ndarray,
    cluster2_data: np.ndarray,
    centroid1: np.ndarray,
    centroid2: np.ndarray,
    density_config: DensityEstimationConfig,
    use_lda_for_merge_test: bool,
    profile_timing: dict
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Test whether two clusters should be merged.

    Projects data onto optimal discriminative direction using LDA,
    then performs 1D separation test.

    Returns:
    --------
    separation_score : float
        Score indicating how separated the clusters are
    cutpoint : float
        Point to split the data if keeping separate
    new_assignments : np.ndarray
        Array of 0s and 1s indicating cluster assignment for each point
    cluster1_1d : np.ndarray
        Projected 1D data for cluster 1
    cluster2_1d : np.ndarray
        Projected 1D data for cluster 2
    """
    n1 = len(cluster1_data)
    n2 = len(cluster2_data)
    n_samples = n1 + n2

    def do_non_lda_fallback(c1_data, c2_data):
        with Timer('merge_test_non_lda_fallback', profile_timing):
            # Use simple centroid-based direction
            lda_direction = centroid2 - centroid1

            # Normalize the direction
            norm = np.linalg.norm(lda_direction)
            if norm > 0:
                lda_direction = lda_direction / norm
            else:
                # Clusters are at the same location, use arbitrary direction
                lda_direction = np.zeros(c1_data.shape[1])
                lda_direction[0] = 1.0

            # Project data manually (optimized - avoid vstack and mean calculation)
            with Timer('merge_test_non_lda_projection', profile_timing):
                cluster1_1d = np.dot(c1_data, lda_direction)
                cluster2_1d = np.dot(c2_data, lda_direction)

            # Perform 1D separation test and get redistribution assignments
            with Timer('merge_test_1d', profile_timing):
                separation_score, cutpoint, new_assignments = _merge_test_1d(
                    cluster1_1d, cluster2_1d, density_config, profile_timing
                )

            return separation_score, cutpoint, new_assignments, cluster1_1d, cluster2_1d

    # Check if we have enough samples for LDA (need at least n_classes + 1)
    if n_samples <= 2 or not use_lda_for_merge_test:
        # Not enough samples for LDA, use simple centroid-based direction
        return do_non_lda_fallback(cluster1_data, cluster2_data)
    else:
        # Use LDA to find optimal discriminative direction
        # Combine data more efficiently
        combined_data = np.vstack([cluster1_data, cluster2_data])
        combined_labels = np.zeros(n_samples, dtype=int)
        combined_labels[:n1] = 0
        combined_labels[n1:] = 1

        try:
            with Timer('merge_test_lda_fit', profile_timing):
                lda = LinearDiscriminantAnalysis(
                    n_components=1, solver="svd", store_covariance=False
                )
                lda.fit(combined_data, combined_labels)
        except Exception as e:
            # lda can fail if covariance matrix is singular
            return do_non_lda_fallback(cluster1_data, cluster2_data)

        # Project data to 1D more efficiently
        with Timer('merge_test_lda_transform', profile_timing):
            projected_data = lda.transform(combined_data).ravel()
        cluster1_1d = projected_data[:n1]
        cluster2_1d = projected_data[n1:]

        # Perform 1D separation test and get redistribution assignments
        with Timer('merge_test_1d', profile_timing):
            separation_score, cutpoint, new_assignments = _merge_test_1d(
                cluster1_1d, cluster2_1d, density_config, profile_timing
            )

        return separation_score, cutpoint, new_assignments, cluster1_1d, cluster2_1d

def _merge_test_1d(
    cluster1_1d: np.ndarray,
    cluster2_1d: np.ndarray,
    density_config: DensityEstimationConfig,
    profile_timing: dict
) -> Tuple[float, float, np.ndarray]:
    """
    Perform 1D separation test between two clusters and redistribute points.

    Algorithm:
    1. Combine points (keep original order)
    2. Find projected centroids
    3. Estimate densities at centroids and find minimum density point between them
    4. Calculate separation score (ratio of endpoint densities to minimum density)
    5. Redistribute points based on cutpoint

    Returns:
    --------
    separation_score : float
        Score indicating how separated the clusters are (higher = more separated)
    cutpoint : float
        Point of minimum density between centroids
    new_assignments : np.ndarray
        Array of 0s and 1s indicating cluster assignment for each point
        (0 = first cluster, 1 = second cluster, in original order [cluster1_1d, cluster2_1d])
    """
    # Combine all points (keep original order for assignment)
    all_points = np.concatenate([cluster1_1d, cluster2_1d])

    # Calculate projected centroids
    a = np.mean(cluster1_1d)
    b = np.mean(cluster2_1d)

    # Estimate density dip using configured method
    with Timer('estimate_density_dip', profile_timing):
        xx = estimate_density_dip(
            all_points, min(a, b), max(a, b), density_config
        )
    density_left = xx["density_a"]
    density_right = xx["density_b"]
    c = xx["c"]
    density_c = xx["density_c"]

    separation_score = (
        min(density_left, density_right) / density_c if density_c > 0 else np.inf
    )
    cutpoint = c

    if a < b:
        new_assignments = np.where(all_points < cutpoint, 0, 1)
    else:
        new_assignments = np.where(all_points > cutpoint, 0, 1)

    return separation_score, cutpoint, new_assignments

def _reassign_labels(labels: np.ndarray) -> np.ndarray:
    """
    Reassign labels to be consecutive starting from 1 with no gaps.

    Example: [1, 5, 5, 9, 1] -> [1, 2, 2, 3, 1]

    Returns:
    --------
    new_labels : np.ndarray
        Labels reassigned to consecutive integers starting from 1
    """
    unique_labels = np.unique(labels)
    # Use searchsorted for vectorized mapping instead of list comprehension
    new_labels = np.searchsorted(unique_labels, labels) + 1
    return new_labels
