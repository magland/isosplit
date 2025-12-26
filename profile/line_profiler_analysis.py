"""
Line-by-line profiling for critical IsoSplit functions.

This script uses line_profiler to identify bottlenecks at the line level
within the most time-consuming functions.
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
    Main function for line profiling.
    
    To run this script with line_profiler:
    
    1. Install line_profiler:
       pip install line_profiler
    
    2. Run with kernprof:
       kernprof -l -v profile/line_profiler_analysis.py
    
    This will generate line_profiler_analysis.py.lprof and display results.
    """
    print("Generating test data...")
    X = generate_test_data(n_clusters=10, samples_per_cluster=1000, n_dimensions=10)
    print(f"Data shape: {X.shape}")
    
    print("\nRunning IsoSplit with line profiling...")
    print("Note: This will be slower than normal execution due to profiling overhead.")
    
    labels = isosplit(X)
    
    print(f"\nClustering complete!")
    print(f"Number of clusters found: {len(np.unique(labels))}")
    print("\nTo profile specific functions, add @profile decorator to them in isosplit/core.py")
    print("Then run: kernprof -l -v profile/line_profiler_analysis.py")

if __name__ == "__main__":
    main()
