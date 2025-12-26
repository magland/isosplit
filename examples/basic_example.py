"""
Basic example demonstrating the IsoSplit clustering algorithm.

This example generates 2D data from three Gaussian clusters and applies
the IsoSplit algorithm to automatically detect and label the clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from isosplit import isosplit

# Set random seed for reproducibility
np.random.seed(42)

# Generate three 2D Gaussian clusters
n_samples_per_cluster = 100

# Cluster 1: centered at (0, 0)
cluster1 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([0, 0])

# Cluster 2: centered at (3, 3)
cluster2 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([3, 3])

# Cluster 3: centered at (3, -3)
cluster3 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([3, -3])

# Combine all clusters
X = np.vstack([cluster1, cluster2, cluster3])

print(f"Input data shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of dimensions: {X.shape[1]}")

# Apply IsoSplit clustering
# Explicitly set optional parameters:
# - separation_threshold: Controls how separated clusters must be to remain separate (default=2)
# - initial_k: Number of initial clusters for k-means (default=30)
labels = isosplit(X, separation_threshold=2.0, initial_k=30)

print(f"\nOutput labels shape: {labels.shape}")
print(f"Number of detected clusters: {len(np.unique(labels))}")
print(f"Unique labels: {np.unique(labels)}")

# Visualize the results
plt.figure(figsize=(10, 5))

# Plot original data (true clusters)
plt.subplot(1, 2, 1)
plt.scatter(cluster1[:, 0], cluster1[:, 1], alpha=0.6, label='Cluster 1')
plt.scatter(cluster2[:, 0], cluster2[:, 1], alpha=0.6, label='Cluster 2')
plt.scatter(cluster3[:, 0], cluster3[:, 1], alpha=0.6, label='Cluster 3')
plt.title('Original Data (True Clusters)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot IsoSplit results
plt.subplot(1, 2, 2)
for label in np.unique(labels):
    mask = labels == label
    plt.scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Cluster {label}')
plt.title('IsoSplit Clustering Results')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
