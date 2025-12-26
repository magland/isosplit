"""
Benchmark scaling behavior of IsoSplit algorithm.

This script tests how the algorithm's performance scales with:
- Number of samples
- Number of dimensions
- Number of clusters
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isosplit import isosplit

def generate_test_data(n_clusters=10, samples_per_cluster=1000, n_dimensions=10):
    """Generate test data for benchmarking."""
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

def benchmark_sample_scaling():
    """Test how performance scales with number of samples."""
    print("\n" + "="*80)
    print("BENCHMARK: Scaling with Number of Samples")
    print("="*80 + "\n")
    
    n_clusters = 10
    n_dimensions = 10
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    
    times = []
    
    for samples_per_cluster in sample_sizes:
        total_samples = n_clusters * samples_per_cluster
        print(f"Testing {total_samples} samples ({samples_per_cluster} per cluster)...")
        
        X = generate_test_data(n_clusters, samples_per_cluster, n_dimensions)
        
        start = time.time()
        labels = isosplit(X)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  Time: {elapsed:.4f}s, Clusters found: {len(np.unique(labels))}")
    
    # Plot results
    total_samples = [n_clusters * s for s in sample_sizes]
    plt.figure(figsize=(10, 6))
    plt.plot(total_samples, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Total Number of Samples', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('IsoSplit Scaling: Number of Samples', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('profile/scaling_samples.png', dpi=150)
    print("\nPlot saved to: profile/scaling_samples.png")
    
    return total_samples, times

def benchmark_dimension_scaling():
    """Test how performance scales with number of dimensions."""
    print("\n" + "="*80)
    print("BENCHMARK: Scaling with Number of Dimensions")
    print("="*80 + "\n")
    
    n_clusters = 10
    samples_per_cluster = 500
    dimensions = [2, 5, 10, 20, 50, 100]
    
    times = []
    
    for n_dimensions in dimensions:
        print(f"Testing {n_dimensions} dimensions...")
        
        X = generate_test_data(n_clusters, samples_per_cluster, n_dimensions)
        
        start = time.time()
        labels = isosplit(X)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  Time: {elapsed:.4f}s, Clusters found: {len(np.unique(labels))}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, times, 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('IsoSplit Scaling: Number of Dimensions', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('profile/scaling_dimensions.png', dpi=150)
    print("\nPlot saved to: profile/scaling_dimensions.png")
    
    return dimensions, times

def benchmark_cluster_scaling():
    """Test how performance scales with number of clusters."""
    print("\n" + "="*80)
    print("BENCHMARK: Scaling with Number of Clusters")
    print("="*80 + "\n")
    
    samples_per_cluster = 500
    n_dimensions = 10
    cluster_counts = [3, 5, 10, 15, 20, 30]
    
    times = []
    
    for n_clusters in cluster_counts:
        print(f"Testing {n_clusters} clusters...")
        
        X = generate_test_data(n_clusters, samples_per_cluster, n_dimensions)
        
        start = time.time()
        labels = isosplit(X)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  Time: {elapsed:.4f}s, Clusters found: {len(np.unique(labels))}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, times, '^-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('IsoSplit Scaling: Number of Clusters', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('profile/scaling_clusters.png', dpi=150)
    print("\nPlot saved to: profile/scaling_clusters.png")
    
    return cluster_counts, times

def main():
    """Run all benchmarks."""
    print("IsoSplit Scaling Benchmarks")
    print("="*80)
    
    # Run benchmarks
    samples, sample_times = benchmark_sample_scaling()
    dims, dim_times = benchmark_dimension_scaling()
    clusters, cluster_times = benchmark_cluster_scaling()
    
    # Create combined summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(samples, sample_times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Total Samples')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_title('Sample Scaling')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(dims, dim_times, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Dimensions')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_title('Dimension Scaling')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(clusters, cluster_times, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Clusters')
    axes[2].set_ylabel('Time (s)')
    axes[2].set_title('Cluster Scaling')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('profile/scaling_summary.png', dpi=150)
    print("\n" + "="*80)
    print("Summary plot saved to: profile/scaling_summary.png")
    print("="*80)
    
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()
