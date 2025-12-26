"""
Profiling script for IsoSplit algorithm.

This script profiles the IsoSplit algorithm to identify performance bottlenecks
using Python's cProfile and line_profiler tools.
"""

import numpy as np
import cProfile
import pstats
import io
from pstats import SortKey
import time
import sys
import os

# Add parent directory to path to import isosplit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isosplit import isosplit


def generate_test_data(n_clusters=10, samples_per_cluster=1000, n_dimensions=10, separation=3.0, random_state=42):
    """
    Generate synthetic test data with multiple well-separated clusters.
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters to generate
    samples_per_cluster : int
        Number of samples per cluster
    n_dimensions : int
        Number of dimensions for the data
    separation : float
        How far apart the cluster centers should be
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : np.ndarray
        Generated data of shape (n_clusters * samples_per_cluster, n_dimensions)
    true_labels : np.ndarray
        True cluster labels
    """
    np.random.seed(random_state)
    
    clusters = []
    true_labels = []
    
    for i in range(n_clusters):
        # Generate cluster center on a grid in high-dimensional space
        center = np.zeros(n_dimensions)
        # Spread centers in first few dimensions
        dim_idx = i % n_dimensions
        center[dim_idx] = (i // n_dimensions) * separation
        
        # Generate cluster data
        cluster = np.random.randn(samples_per_cluster, n_dimensions) * 0.5 + center
        clusters.append(cluster)
        true_labels.extend([i + 1] * samples_per_cluster)
    
    X = np.vstack(clusters)
    true_labels = np.array(true_labels)
    
    return X, true_labels


def profile_with_cprofile(X, output_file='profile_results.txt'):
    """
    Profile the isosplit function using cProfile.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data to cluster
    output_file : str
        File to save profiling results
    """
    print(f"\n{'='*80}")
    print(f"Running cProfile on data with shape {X.shape}")
    print(f"{'='*80}\n")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the algorithm
    start_time = time.time()
    labels = isosplit(X)
    end_time = time.time()
    
    profiler.disable()
    
    # Print results to console
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats(SortKey.CUMULATIVE)
    
    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80 + "\n")
    ps.print_stats(30)
    print(s.getvalue())
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(f"IsoSplit Profiling Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Data shape: {X.shape}\n")
        f.write(f"Total execution time: {end_time - start_time:.4f} seconds\n")
        f.write(f"Number of clusters found: {len(np.unique(labels))}\n\n")
        f.write("="*80 + "\n")
        f.write("TOP FUNCTIONS BY CUMULATIVE TIME\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
    
    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
    print(f"Number of clusters found: {len(np.unique(labels))}")
    print(f"\nDetailed results saved to: {output_file}")
    
    return labels


def time_individual_components(X, n_runs=3):
    """
    Time individual components of the algorithm separately.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data to cluster
    n_runs : int
        Number of runs to average over
    """
    print(f"\n{'='*80}")
    print(f"Timing Individual Components (averaged over {n_runs} runs)")
    print(f"{'='*80}\n")
    
    times = []
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}...")
        start = time.time()
        labels = isosplit(X)
        total_time = time.time() - start
        times.append(total_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nAverage total time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Min time: {min(times):.4f} seconds")
    print(f"Max time: {max(times):.4f} seconds")
    
    return avg_time


def main():
    """Main profiling routine."""
    print("IsoSplit Performance Profiling")
    print("="*80)
    
    # Test configurations: (n_clusters, samples_per_cluster, n_dimensions, separation)
    test_configs = [
        (5, 500, 5, 3.0, "Small: 5 clusters, 2500 samples, 5D"),
        (10, 1000, 10, 3.0, "Medium: 10 clusters, 10000 samples, 10D"),
        (15, 1000, 10, 3.0, "Large: 15 clusters, 15000 samples, 10D"),
    ]
    
    results_summary = []
    
    for i, (n_clusters, samples_per_cluster, n_dimensions, separation, description) in enumerate(test_configs):
        print(f"\n\n{'#'*80}")
        print(f"TEST CONFIGURATION {i+1}: {description}")
        print(f"{'#'*80}")
        
        # Generate test data
        print(f"\nGenerating test data...")
        X, true_labels = generate_test_data(
            n_clusters=n_clusters,
            samples_per_cluster=samples_per_cluster,
            n_dimensions=n_dimensions,
            separation=separation,
            random_state=42
        )
        
        print(f"Data shape: {X.shape}")
        print(f"True number of clusters: {n_clusters}")
        
        # Profile with cProfile
        output_file = f'profile_results_{i+1}.txt'
        labels = profile_with_cprofile(X, output_file)
        
        # Time individual components
        avg_time = time_individual_components(X, n_runs=3)
        
        results_summary.append({
            'description': description,
            'shape': X.shape,
            'true_clusters': n_clusters,
            'found_clusters': len(np.unique(labels)),
            'avg_time': avg_time,
        })
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*80}\n")
    
    for result in results_summary:
        print(f"{result['description']}")
        print(f"  Shape: {result['shape']}")
        print(f"  True clusters: {result['true_clusters']}, Found: {result['found_clusters']}")
        print(f"  Average time: {result['avg_time']:.4f} seconds")
        print()
    
    print("\nProfiling complete! Check profile_results_*.txt for detailed breakdowns.")


if __name__ == "__main__":
    main()
