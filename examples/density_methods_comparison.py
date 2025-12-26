"""
Example demonstrating different density estimation methods in IsoSplit.

This example compares the performance and results of:
- GMM (Gaussian Mixture Model) - default, slowest but most flexible
- KDE (Kernel Density Estimation) - fast, good balance
- Histogram - fastest, less smooth

Author: IsoSplit Contributors
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from isosplit import isosplit, DensityEstimationConfig

# Set random seed for reproducibility
np.random.seed(42)

# Generate test data with multiple clusters
print("Generating test data...")
n_clusters = 10
samples_per_cluster = 1000
n_dimensions = 10

clusters = []
# Create clusters in a grid pattern in the first 2 dimensions
grid_size = int(np.ceil(np.sqrt(n_clusters)))
for i in range(n_clusters):
    center = np.zeros(n_dimensions)
    # Place clusters in a grid for first 2 dimensions (for visualization)
    row = i // grid_size
    col = i % grid_size
    center[0] = col * 4.0  # Spread in dimension 1
    center[1] = row * 4.0  # Spread in dimension 2
    # Add variation in higher dimensions
    for d in range(2, n_dimensions):
        center[d] = np.random.randn() * 2.0
    
    cluster = np.random.randn(samples_per_cluster, n_dimensions) * 0.5 + center
    clusters.append(cluster)

X = np.vstack(clusters)
print(f"Data shape: {X.shape}")
print(f"True number of clusters: {n_clusters}\n")

# Test different density estimation methods
methods = [
    ('GMM (default)', DensityEstimationConfig(method='gmm')),
    ('KDE', DensityEstimationConfig(method='kde')),
    ('Histogram', DensityEstimationConfig(method='histogram')),
]

results = []

print("="*80)
print("COMPARING DENSITY ESTIMATION METHODS")
print("="*80)

for name, config in methods:
    print(f"\n{name}:")
    print("-" * 40)
    
    # Time the clustering
    start_time = time.time()
    labels = isosplit(X, density_config=config)
    elapsed_time = time.time() - start_time
    
    n_clusters_found = len(np.unique(labels))
    
    print(f"  Time: {elapsed_time:.4f} seconds")
    print(f"  Clusters found: {n_clusters_found}")
    
    results.append({
        'name': name,
        'config': config,
        'time': elapsed_time,
        'n_clusters': n_clusters_found,
        'labels': labels
    })

# Print summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"{'Method':<20} {'Time (s)':<15} {'Clusters':<15} {'Speedup':<15}")
print("-" * 80)

baseline_time = results[0]['time']
for r in results:
    speedup = baseline_time / r['time']
    print(f"{r['name']:<20} {r['time']:<15.4f} {r['n_clusters']:<15} {speedup:<15.2f}x")

# Check if results are consistent across methods
print("\n" + "="*80)
print("CONSISTENCY CHECK")
print("="*80)

# Compare cluster counts
cluster_counts = [r['n_clusters'] for r in results]
if len(set(cluster_counts)) == 1:
    print(f"✓ All methods found the same number of clusters: {cluster_counts[0]}")
else:
    print(f"⚠ Methods found different numbers of clusters: {cluster_counts}")

# Create visualization comparing the methods
print("\n" + "="*80)
print("CREATING VISUALIZATIONS...")
print("="*80)

# Create scatter plots for first 2 dimensions
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

# Plot true clusters
ax = axes[0]
true_labels = np.repeat(np.arange(1, n_clusters + 1), samples_per_cluster)
scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='tab10', alpha=0.6, s=20)
ax.set_title('True Clusters', fontsize=14, fontweight='bold')
ax.set_xlabel('Dimension 1', fontsize=11)
ax.set_ylabel('Dimension 2', fontsize=11)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster ID')

# Plot results from each method
for idx, result in enumerate(results):
    ax = axes[idx + 1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='tab10', alpha=0.6, s=20)
    title = f"{result['name']}\n({result['time']:.3f}s, {result['n_clusters']} clusters)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

plt.tight_layout()
plt.savefig('examples/density_methods_comparison.png', dpi=150, bbox_inches='tight')
print("Saved scatter plot comparison to: examples/density_methods_comparison.png")

# Create performance comparison bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Execution time comparison
method_names = [r['name'] for r in results]
times = [r['time'] for r in results]
# Use a color for each method
colors = ['#ff6b6b', '#4ecdc4', '#95e1d3'][:len(results)]

bars = ax1.bar(method_names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time_val:.3f}s',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup comparison
speedups = [baseline_time / t for t in times]
bars = ax2.bar(method_names, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup Factor (vs GMM)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (GMM)')
ax2.grid(True, axis='y', alpha=0.3)
ax2.legend(fontsize=11)

# Add value labels on bars
for bar, speedup in zip(bars, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup:.2f}x',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('examples/density_methods_performance.png', dpi=150, bbox_inches='tight')
print("Saved performance comparison to: examples/density_methods_performance.png")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
1. GMM (Gaussian Mixture Model):
   - Use when: You need maximum flexibility and accuracy
   - Pros: Most sophisticated, handles complex distributions
   - Cons: Slowest method
   - Best for: Small to medium datasets, research applications

2. KDE (Kernel Density Estimation):
   - Use when: You want a good balance of speed and accuracy
   - Pros: Much faster than GMM, still quite accurate
   - Cons: May be less flexible than GMM
   - Best for: General purpose clustering, medium to large datasets
   - RECOMMENDED for most use cases

3. Histogram:
   - Use when: Speed is critical and you have large datasets
   - Pros: Fastest method, simple and robust
   - Cons: May be less smooth, bin selection matters
   - Best for: Very large datasets, real-time applications
""")

# Example: Fine-tuning parameters
print("\n" + "="*80)
print("EXAMPLE: FINE-TUNING PARAMETERS")
print("="*80)

# KDE with custom bandwidth
print("\nKDE with custom bandwidth:")
config_kde_custom = DensityEstimationConfig(
    method='kde',
    kde_bandwidth=0.5  # Custom bandwidth
)
start = time.time()
labels = isosplit(X, density_config=config_kde_custom)
elapsed = time.time() - start
print(f"  Time: {elapsed:.4f}s, Clusters: {len(np.unique(labels))}")

# Histogram with more bins
print("\nHistogram with 100 bins:")
config_hist_fine = DensityEstimationConfig(
    method='histogram',
    hist_n_bins=100,
    hist_smoothing=True,
    hist_smoothing_sigma=2.0
)
start = time.time()
labels = isosplit(X, density_config=config_hist_fine)
elapsed = time.time() - start
print(f"  Time: {elapsed:.4f}s, Clusters: {len(np.unique(labels))}")

# GMM with fewer components for speed
print("\nGMM with max 3 components (faster):")
config_gmm_fast = DensityEstimationConfig(
    method='gmm',
    gmm_max_components=3  # Reduce from default 5
)
start = time.time()
labels = isosplit(X, density_config=config_gmm_fast)
elapsed = time.time() - start
print(f"  Time: {elapsed:.4f}s, Clusters: {len(np.unique(labels))}")

print("\n" + "="*80)
print("Comparison complete!")
print("Plots saved to examples/ directory")
print("="*80)
