"""
Memory profiling for IsoSplit algorithm.

This script uses memory_profiler to track memory usage and identify
memory bottlenecks in the algorithm.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isosplit import isosplit

def generate_test_data(n_clusters=10, samples_per_cluster=1000, n_dimensions=10):
    """Generate test data for profiling."""
    np.random.seed(42)
    
    clusters = []
    for i in range(n_clusters):
        center = np.zeros(n_dimensions)
        dim_idx = i % n_dimensions
        center[dim_idx] = (i // n_dimensions) * 3.0
        cluster = np.random.randn(samples_per_cluster, n_dimensions) * 0.5 + center
        clusters.append(cluster)
    
    X = np.vstack(clusters)
    return X

def main():
    """
    Main function for memory profiling.
    
    To run this script with memory_profiler:
    
    1. Install memory_profiler:
       pip install memory_profiler
    
    2. Run with:
       python -m memory_profiler profile/memory_profiler_analysis.py
    
    For line-by-line memory profiling, add @profile decorator to functions
    in isosplit/core.py and run the same command.
    """
    print("Generating test data...")
    X = generate_test_data(n_clusters=10, samples_per_cluster=1000, n_dimensions=10)
    print(f"Data shape: {X.shape}")
    print(f"Data memory size: {X.nbytes / 1024 / 1024:.2f} MB")
    
    print("\nRunning IsoSplit with memory profiling...")
    
    labels = isosplit(X)
    
    print(f"\nClustering complete!")
    print(f"Number of clusters found: {len(np.unique(labels))}")

if __name__ == "__main__":
    main()
