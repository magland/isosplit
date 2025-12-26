"""
Verbose mode utilities for isosplit_next clustering algorithm.
"""
import numpy as np
import os

# Check for verbose mode
VERBOSE = os.environ.get('ISOSPLIT_VERBOSE', '0') == '1'

if VERBOSE:
    import matplotlib.pyplot as plt

PLOT_COUNTER = 0

def print_header(initial_k: int, separation_threshold: float, data_shape: tuple) -> None:
    """Print initial header for verbose mode."""
    if not VERBOSE:
        return
    print(f"\n{'='*60}")
    print(f"ISOSPLIT VERBOSE MODE")
    print(f"{'='*60}")
    print(f"Initial k-means clustering: {initial_k} clusters")
    print(f"Separation threshold: {separation_threshold}")
    print(f"Data shape: {data_shape}")

def print_iteration_info(passnum: int, iteration: int, cluster1_id: int, cluster2_id: int,
                         cluster1_size: int, cluster2_size: int) -> None:
    """Print information about current iteration."""
    if not VERBOSE:
        return
    print(f"\n{'-'*60}")
    print(f"Iteration {iteration} (pass {passnum}): Testing clusters {cluster1_id} & {cluster2_id}")
    print(f"Cluster {cluster1_id} size: {cluster1_size}")
    print(f"Cluster {cluster2_id} size: {cluster2_size}")

def print_separation_info(separation_score: float, threshold: float) -> None:
    """Print separation score and threshold."""
    if not VERBOSE:
        return
    print(f"Separation score: {separation_score:.4f}")
    print(f"Threshold: {threshold:.4f}")

def print_decision_merge(merge_to: int) -> None:
    """Print merge decision."""
    if not VERBOSE:
        return
    print(f"Decision: MERGE into cluster {merge_to}")

def print_decision_redistribute(changed: int) -> None:
    """Print redistribute decision."""
    if not VERBOSE:
        return
    print(f"Decision: KEEP SEPARATE & REDISTRIBUTE")
    print(f"Points redistributed: {changed}")

def plot_clusters(data: np.ndarray, labels: np.ndarray, title: str, iteration: int, passnum: int) -> None:
    """Plot clusters (only in verbose mode, only works for 2D or 3D data)."""
    if not VERBOSE:
        return
    
    global PLOT_COUNTER
    PLOT_COUNTER += 1
    
    n_features = data.shape[1]
    
    if n_features == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"{title}\n{len(np.unique(labels))} clusters")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'verbose_{PLOT_COUNTER:03d}_iter{iteration:03d}_pass{passnum}_{title.replace(" ", "_")}.png', dpi=100)
        plt.close()
    elif n_features == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab20', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster')
        ax.set_title(f"{title}\n{len(np.unique(labels))} clusters")
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.tight_layout()
        plt.savefig(f'verbose_{PLOT_COUNTER:03d}_iter{iteration:03d}_pass{passnum}_{title.replace(" ", "_")}.png', dpi=100)
        plt.close()
    else:
        # For higher dimensions, just print info
        print(f"  [Visualization skipped: {n_features}D data]")

def plot_1d_histogram(cluster1_1d: np.ndarray, cluster2_1d: np.ndarray, 
                      cutpoint: float, separation_score: float,
                      cluster1_id: int, cluster2_id: int, 
                      decision: str, iteration: int, passnum: int) -> None:
    """Plot 1D histograms showing projected data with cutpoint and separation score."""
    if not VERBOSE:
        return
    
    global PLOT_COUNTER
    PLOT_COUNTER += 1
    
    plt.figure(figsize=(10, 6))
    
    # Determine histogram bins
    all_data = np.concatenate([cluster1_1d, cluster2_1d])
    bins = np.linspace(all_data.min(), all_data.max(), int(np.sqrt(len(all_data))) + 1)
    
    # Plot histograms
    plt.hist(cluster1_1d, bins=bins, alpha=0.6, label=f'Cluster {cluster1_id}', color='red', edgecolor='black')
    plt.hist(cluster2_1d, bins=bins, alpha=0.6, label=f'Cluster {cluster2_id}', color='blue', edgecolor='black')
    
    # Plot cutpoint as dashed vertical line
    plt.axvline(cutpoint, color='green', linestyle='--', linewidth=2, label=f'Cutpoint: {cutpoint:.3f}')
    
    # Add labels and title
    plt.xlabel('Projected Value (LDA direction)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'1D Projection - Iteration {iteration}\n'
              f'Separation Score: {separation_score:.4f} | Decision: {decision}',
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'verbose_{PLOT_COUNTER:03d}_iter{iteration:03d}_pass{passnum}_histogram_{decision}_c{cluster1_id}_c{cluster2_id}.png', dpi=100)
    plt.close()

    # Also save the info to .npy file
    np.save(f'verbose_{PLOT_COUNTER:03d}_iter{iteration:03d}_pass{passnum}_histogram_{decision}_c{cluster1_id}_c{cluster2_id}.npy', {
        'cluster1_1d': cluster1_1d,
        'cluster2_1d': cluster2_1d,
        'cutpoint': cutpoint,
        'separation_score': separation_score,
        'decision': decision
    })

def plot_decision(data: np.ndarray, old_labels: np.ndarray, new_labels: np.ndarray,
                  cluster1_id: int, cluster2_id: int, decision: str, iteration: int, passnum: int) -> None:
    """Plot before/after comparison for a merge or redistribute decision (verbose mode only)."""
    if not VERBOSE:
        return
    
    global PLOT_COUNTER
    PLOT_COUNTER += 1
    
    n_features = data.shape[1]
    
    # Only visualize for 2D data
    if n_features != 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Highlight the two clusters being compared
    mask_clusters = (old_labels == cluster1_id) | (old_labels == cluster2_id)
    
    # Before
    ax1.scatter(data[~mask_clusters, 0], data[~mask_clusters, 1], 
                c='lightgray', alpha=0.3, s=20, label='Other clusters')
    ax1.scatter(data[old_labels == cluster1_id, 0], data[old_labels == cluster1_id, 1],
                c='red', alpha=0.7, s=40, label=f'Cluster {cluster1_id}')
    ax1.scatter(data[old_labels == cluster2_id, 0], data[old_labels == cluster2_id, 1],
                c='blue', alpha=0.7, s=40, label=f'Cluster {cluster2_id}')
    ax1.set_title(f'Before: Clusters {cluster1_id} & {cluster2_id}')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # After
    mask_clusters_new = (new_labels == cluster1_id) | (new_labels == cluster2_id)
    ax2.scatter(data[~mask_clusters_new, 0], data[~mask_clusters_new, 1],
                c='lightgray', alpha=0.3, s=20, label='Other clusters')
    
    if decision == "Merged":
        merged_id = min(cluster1_id, cluster2_id)
        ax2.scatter(data[new_labels == merged_id, 0], data[new_labels == merged_id, 1],
                    c='purple', alpha=0.7, s=40, label=f'Merged → {merged_id}')
    else:  # Redistributed
        ax2.scatter(data[new_labels == cluster1_id, 0], data[new_labels == cluster1_id, 1],
                    c='red', alpha=0.7, s=40, label=f'Cluster {cluster1_id}')
        ax2.scatter(data[new_labels == cluster2_id, 0], data[new_labels == cluster2_id, 1],
                    c='blue', alpha=0.7, s=40, label=f'Cluster {cluster2_id}')
    
    ax2.set_title(f'After: {decision}')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'Iteration {iteration}: {decision}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'verbose_{PLOT_COUNTER:03d}_iter{iteration:03d}_pass{passnum}_{decision}_c{cluster1_id}_c{cluster2_id}.png', dpi=100)
    plt.close()
