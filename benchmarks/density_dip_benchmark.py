"""
Benchmark for estimate_density_dip function.

This script evaluates the accuracy of estimate_density_dip across different
density estimation methods (GMM, KDE, histogram) using a mixture of two Gaussians.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import sys
import os
import argparse
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isosplit.density_estimation import estimate_density_dip, DensityEstimationConfig


def generate_gaussian_mixture(n1, n2, mu1, mu2, sigma1, sigma2, random_seed=42):
    """
    Generate a sample from a mixture of two Gaussians.
    
    Parameters
    ----------
    n1, n2 : int
        Number of samples from each Gaussian
    mu1, mu2 : float
        Means of the two Gaussians
    sigma1, sigma2 : float
        Standard deviations of the two Gaussians
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    sample : ndarray
        Combined sample from both Gaussians
    """
    np.random.seed(random_seed)
    
    sample1 = np.random.normal(mu1, sigma1, n1)
    sample2 = np.random.normal(mu2, sigma2, n2)
    
    sample = np.concatenate([sample1, sample2])
    np.random.shuffle(sample)
    
    return sample


def theoretical_density(x, n1, n2, mu1, mu2, sigma1, sigma2):
    """
    Compute theoretical density of mixture of two Gaussians at point(s) x.
    
    The mixture density is:
    f(x) = w1 * N(x; mu1, sigma1) + w2 * N(x; mu2, sigma2)
    
    where w1 = n1/(n1+n2) and w2 = n2/(n1+n2)
    """
    n_total = n1 + n2
    w1 = n1 / n_total
    w2 = n2 / n_total
    
    density1 = norm.pdf(x, mu1, sigma1)
    density2 = norm.pdf(x, mu2, sigma2)
    
    return w1 * density1 + w2 * density2


def find_theoretical_minimum(a, b, n1, n2, mu1, mu2, sigma1, sigma2):
    """
    Find the theoretical minimum density point between a and b.
    
    Returns
    -------
    c_true : float
        Point of minimum density
    density_c_true : float
        Theoretical density at c_true
    """
    result = minimize_scalar(
        lambda x: theoretical_density(x, n1, n2, mu1, mu2, sigma1, sigma2),
        bounds=(a, b),
        method='bounded'
    )
    
    c_true = result.x
    density_c_true = theoretical_density(c_true, n1, n2, mu1, mu2, sigma1, sigma2)
    
    return c_true, density_c_true


def run_single_trial(n1, n2, mu1, mu2, sigma1, sigma2, random_seed, visualize=False):
    """
    Run a single trial of the benchmark.
    
    Parameters
    ----------
    n1, n2 : int
        Number of samples from each Gaussian
    mu1, mu2 : float
        Means of the two Gaussians (mu1 < mu2)
    sigma1, sigma2 : float
        Standard deviations of the two Gaussians
    random_seed : int
        Random seed for this trial
    visualize : bool
        Whether to create visualization for this trial
        
    Returns
    -------
    trial_results : dict
        Results for this trial including errors for each method
    """
    # Generate sample
    sample = generate_gaussian_mixture(n1, n2, mu1, mu2, sigma1, sigma2, random_seed)
    
    # Set a and b at the means
    a = mu1
    b = mu2
    
    # Calculate theoretical values
    c_true, density_c_true = find_theoretical_minimum(a, b, n1, n2, mu1, mu2, sigma1, sigma2)
    density_a_true = theoretical_density(a, n1, n2, mu1, mu2, sigma1, sigma2)
    density_b_true = theoretical_density(b, n1, n2, mu1, mu2, sigma1, sigma2)
    
    print(f"Theoretical values:")
    print(f"  c_true = {c_true:.4f}")
    print(f"  density(a) = {density_a_true:.6f}")
    print(f"  density(b) = {density_b_true:.6f}")
    print(f"  density(c) = {density_c_true:.6f}")
    print()
    
    # Test all three methods
    methods = ['gmm', 'kde', 'histogram']
    results = {}
    
    for method in methods:
        print(f"Testing method: {method}")
        config = DensityEstimationConfig(method=method)
        
        result = estimate_density_dip(sample, a, b, config)
        results[method] = result
        
        # Calculate errors
        c_error = abs(result['c'] - c_true)
        density_c_error = abs(result['density_c'] - density_c_true)
        relative_error = c_error / (b - a)
        
        print(f"  c = {result['c']:.4f} (error: {c_error:.4f}, relative: {relative_error:.2%})")
        print(f"  density(a) = {result['density_a']:.6f}")
        print(f"  density(b) = {result['density_b']:.6f}")
        print(f"  density(c) = {result['density_c']:.6f} (error: {density_c_error:.6f})")
        print()
    
    # Create visualization only if requested
    if visualize:
        create_visualization(sample, a, b, results, c_true, n1, n2, mu1, mu2, sigma1, sigma2)
    
    # Compile trial results with errors
    trial_results = {
        'c_true': c_true,
        'density_c_true': density_c_true,
    }
    
    for method in methods:
        trial_results[f'{method}_c'] = results[method]['c']
        trial_results[f'{method}_c_error'] = abs(results[method]['c'] - c_true)
        trial_results[f'{method}_c_relative_error'] = abs(results[method]['c'] - c_true) / (b - a)
        trial_results[f'{method}_density_a'] = results[method]['density_a']
        trial_results[f'{method}_density_b'] = results[method]['density_b']
        trial_results[f'{method}_density_c'] = results[method]['density_c']
        trial_results[f'{method}_density_c_error'] = abs(results[method]['density_c'] - density_c_true)
        
        # Calculate separation score (as used in isosplit/core.py)
        density_left = results[method]['density_a']
        density_right = results[method]['density_b']
        density_min = results[method]['density_c']
        separation_score = min(density_left, density_right) / density_min if density_min > 0 else np.inf
        trial_results[f'{method}_separation_score'] = separation_score
    
    return trial_results

def run_benchmark(n1=1000, n2=1000, mu1=0.0, mu2=5.0, sigma1=1.0, sigma2=1.0, n_trials=100):
    """
    Run benchmark comparing three density estimation methods across multiple trials.
    
    Parameters
    ----------
    n1, n2 : int
        Number of samples from each Gaussian
    mu1, mu2 : float
        Means of the two Gaussians (mu1 < mu2)
    sigma1, sigma2 : float
        Standard deviations of the two Gaussians
    n_trials : int
        Number of trials to run
    """
    print("="*80)
    print("DENSITY DIP BENCHMARK")
    print("="*80)
    print(f"\nParameters:")
    print(f"  n1={n1}, n2={n2}")
    print(f"  μ1={mu1}, μ2={mu2}")
    print(f"  σ1={sigma1}, σ2={sigma2}")
    print(f"  Separation: {mu2 - mu1}")
    print(f"  Number of trials: {n_trials}")
    print()
    
    # Store results from all trials
    all_results = []
    
    # Run trials
    for trial in range(n_trials):
        if trial == 0:
            print(f"Running trial {trial + 1}/{n_trials} (with visualization)...")
            trial_result = run_single_trial(n1, n2, mu1, mu2, sigma1, sigma2, 
                                           random_seed=42 + trial, visualize=True)
        else:
            if (trial + 1) % 10 == 0:
                print(f"Running trial {trial + 1}/{n_trials}...")
            trial_result = run_single_trial(n1, n2, mu1, mu2, sigma1, sigma2, 
                                           random_seed=42 + trial, visualize=False)
        
        all_results.append(trial_result)
    
    print()
    
    # Analyze results
    analyze_results(all_results, n1, n2, mu1, mu2, sigma1, sigma2)
    
    return all_results

def analyze_results(all_results, n1, n2, mu1, mu2, sigma1, sigma2):
    """
    Analyze and summarize results across all trials.
    
    Parameters
    ----------
    all_results : list of dict
        Results from all trials
    """
    print("="*80)
    print("SUMMARY STATISTICS ACROSS ALL TRIALS")
    print("="*80)
    print()
    
    methods = ['gmm', 'kde', 'histogram']
    
    # Calculate theoretical context
    c_true_mean = np.mean([r['c_true'] for r in all_results])
    density_c_true_mean = np.mean([r['density_c_true'] for r in all_results])
    separation = mu2 - mu1
    
    # Calculate theoretical densities at a and b
    density_a_true = theoretical_density(mu1, n1, n2, mu1, mu2, sigma1, sigma2)
    density_b_true = theoretical_density(mu2, n1, n2, mu1, mu2, sigma1, sigma2)
    
    # Calculate theoretical separation score
    theoretical_sep_score = min(density_a_true, density_b_true) / density_c_true_mean if density_c_true_mean > 0 else np.inf
    
    print(f"Theoretical Context:")
    print(f"  Separation (μ2 - μ1): {separation:.4f}")
    print(f"  Mean theoretical c: {c_true_mean:.4f}")
    print(f"  Mean theoretical density(c): {density_c_true_mean:.6f}")
    print(f"  Theoretical separation score: {theoretical_sep_score:.4f}")
    print()
    
    # Convert to arrays for easier analysis
    results_arrays = {}
    for method in methods:
        results_arrays[method] = {
            'c_error': np.array([r[f'{method}_c_error'] for r in all_results]),
            'c_relative_error': np.array([r[f'{method}_c_relative_error'] for r in all_results]),
            'density_c_error': np.array([r[f'{method}_density_c_error'] for r in all_results]),
            'c_values': np.array([r[f'{method}_c'] for r in all_results]),
            'density_c_values': np.array([r[f'{method}_density_c'] for r in all_results]),
            'separation_scores': np.array([r[f'{method}_separation_score'] for r in all_results]),
        }
    
    # Print summary statistics with context
    print(f"{'Method':<12} {'Mean c':<12} {'c Error':<12} {'Rel Error':<12} {'Mean dens(c)':<15} {'Dens Error':<15}")
    print("-" * 90)
    
    for method in methods:
        mean_c = np.mean(results_arrays[method]['c_values'])
        mean_c_error = np.mean(results_arrays[method]['c_error'])
        mean_rel_error = np.mean(results_arrays[method]['c_relative_error'])
        mean_density_c = np.mean(results_arrays[method]['density_c_values'])
        mean_density_error = np.mean(results_arrays[method]['density_c_error'])
        
        print(f"{method:<12} {mean_c:<12.4f} {mean_c_error:<12.6f} {mean_rel_error:<12.4%} {mean_density_c:<15.6f} {mean_density_error:<15.6f}")
    
    print()
    print(f"Relative Error Context:")
    print(f"  Errors shown as fraction of separation distance ({separation:.2f})")
    print()
    
    # Create summary visualization
    create_summary_visualization(results_arrays, methods, n1, n2, mu1, mu2, sigma1, sigma2, 
                                 c_true_mean, density_c_true_mean, all_results)

def create_summary_visualization(results_arrays, methods, n1, n2, mu1, mu2, sigma1, sigma2, 
                                 c_true_mean, density_c_true_mean, all_results):
    """
    Create visualization summarizing results across all trials.
    Shows estimated values with theoretical references.
    """
    # Calculate theoretical density at a and b
    density_a_true = theoretical_density(mu1, n1, n2, mu1, mu2, sigma1, sigma2)
    density_b_true = theoretical_density(mu2, n1, n2, mu1, mu2, sigma1, sigma2)
    
    # Collect density_a and density_b values from results
    for method in methods:
        results_arrays[method]['density_a_values'] = np.array([r[f'{method}_density_a'] for r in all_results])
        results_arrays[method]['density_b_values'] = np.array([r[f'{method}_density_b'] for r in all_results])
    
    # Calculate theoretical separation score
    theoretical_sep_score = min(density_a_true, density_b_true) / density_c_true_mean if density_c_true_mean > 0 else np.inf
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    colors = {'gmm': 'green', 'kde': 'orange', 'histogram': 'purple'}
    
    # Plot 1: Estimated c values vs theoretical
    ax1 = axes[0]
    c_vals = [results_arrays[method]['c_values'] for method in methods]
    bp1 = ax1.boxplot(c_vals, labels=methods, patch_artist=True)
    for patch, method in zip(bp1['boxes'], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.6)
    ax1.axhline(c_true_mean, color='black', linestyle='--', linewidth=2, 
                label=f'Theoretical = {c_true_mean:.3f}', alpha=0.7)
    ax1.set_ylabel('Estimated c value', fontsize=12)
    ax1.set_title('Estimated c Values', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Estimated density(a) values vs theoretical
    ax2 = axes[1]
    dens_a_vals = [results_arrays[method]['density_a_values'] for method in methods]
    bp2 = ax2.boxplot(dens_a_vals, labels=methods, patch_artist=True)
    for patch, method in zip(bp2['boxes'], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.6)
    ax2.axhline(density_a_true, color='black', linestyle='--', linewidth=2, 
                label=f'Theoretical = {density_a_true:.4f}', alpha=0.7)
    ax2.set_ylabel('Estimated density(a) value', fontsize=12)
    ax2.set_title('Estimated density(a) Values', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Estimated density(b) values vs theoretical
    ax3 = axes[2]
    dens_b_vals = [results_arrays[method]['density_b_values'] for method in methods]
    bp3 = ax3.boxplot(dens_b_vals, labels=methods, patch_artist=True)
    for patch, method in zip(bp3['boxes'], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.6)
    ax3.axhline(density_b_true, color='black', linestyle='--', linewidth=2, 
                label=f'Theoretical = {density_b_true:.4f}', alpha=0.7)
    ax3.set_ylabel('Estimated density(b) value', fontsize=12)
    ax3.set_title('Estimated density(b) Values', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Estimated density(c) values vs theoretical
    ax4 = axes[3]
    dens_c_vals = [results_arrays[method]['density_c_values'] for method in methods]
    bp4 = ax4.boxplot(dens_c_vals, labels=methods, patch_artist=True)
    for patch, method in zip(bp4['boxes'], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.6)
    ax4.axhline(density_c_true_mean, color='black', linestyle='--', linewidth=2, 
                label=f'Theoretical = {density_c_true_mean:.4f}', alpha=0.7)
    ax4.set_ylabel('Estimated density(c) value', fontsize=12)
    ax4.set_title('Estimated density(c) Values', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Separation scores vs theoretical
    ax5 = axes[4]
    sep_scores = [results_arrays[method]['separation_scores'] for method in methods]
    bp5 = ax5.boxplot(sep_scores, labels=methods, patch_artist=True)
    for patch, method in zip(bp5['boxes'], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.6)
    ax5.axhline(theoretical_sep_score, color='black', linestyle='--', linewidth=2, 
                label=f'Theoretical = {theoretical_sep_score:.2f}', alpha=0.7)
    ax5.set_ylabel('Separation Score', fontsize=12)
    ax5.set_title('Separation Score\n(min(dens_a, dens_b) / dens_c)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Benchmark Summary: n1={n1}, n2={n2}, μ1={mu1}, μ2={mu2}, σ1={sigma1}, σ2={sigma2}',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save figure
    output_path = 'benchmarks/results/density_dip_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary visualization saved to: {output_path}")
    print()


def create_visualization(sample, a, b, results, c_true, n1, n2, mu1, mu2, sigma1, sigma2):
    """
    Create a two-panel visualization showing:
    1. Sample histogram with a, b, c marked
    2. Theoretical vs estimated densities
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Sample histogram with markers
    ax1.hist(sample, bins=50, density=True, alpha=0.6, color='gray', edgecolor='black')
    ax1.axvline(a, color='blue', linestyle='--', linewidth=2, label=f'a = μ1 = {a}')
    ax1.axvline(b, color='red', linestyle='--', linewidth=2, label=f'b = μ2 = {b}')
    
    # Mark c for each method
    colors = {'gmm': 'green', 'kde': 'orange', 'histogram': 'purple'}
    for method, result in results.items():
        ax1.axvline(result['c'], color=colors[method], linestyle=':', 
                   linewidth=2, alpha=0.7, label=f"c ({method}) = {result['c']:.3f}")
    
    ax1.axvline(c_true, color='black', linestyle='-', linewidth=2, 
               alpha=0.5, label=f'c (theoretical) = {c_true:.3f}')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Sample Distribution with Markers', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Theoretical vs estimated densities
    x_range = np.linspace(min(sample), max(sample), 500)
    theoretical = theoretical_density(x_range, n1, n2, mu1, mu2, sigma1, sigma2)
    
    ax2.plot(x_range, theoretical, 'k-', linewidth=2.5, label='Theoretical', alpha=0.8)
    ax2.axvline(a, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(b, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(c_true, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Mark theoretical densities
    ax2.plot(a, theoretical_density(a, n1, n2, mu1, mu2, sigma1, sigma2), 
            'bo', markersize=10, label='density(a) theoretical', zorder=5)
    ax2.plot(b, theoretical_density(b, n1, n2, mu1, mu2, sigma1, sigma2), 
            'ro', markersize=10, label='density(b) theoretical', zorder=5)
    ax2.plot(c_true, theoretical_density(c_true, n1, n2, mu1, mu2, sigma1, sigma2), 
            'ko', markersize=10, label='density(c) theoretical', zorder=5)
    
    # Mark estimated densities for each method
    markers = {'gmm': 's', 'kde': '^', 'histogram': 'D'}
    for method, result in results.items():
        ax2.plot(a, result['density_a'], markers[method], 
                color=colors[method], markersize=8, alpha=0.7,
                label=f"density(a) {method}")
        ax2.plot(b, result['density_b'], markers[method], 
                color=colors[method], markersize=8, alpha=0.7)
        ax2.plot(result['c'], result['density_c'], markers[method], 
                color=colors[method], markersize=8, alpha=0.7,
                label=f"density(c) {method}")
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Theoretical vs Estimated Densities', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'benchmarks/results/density_dip_benchmark.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark estimate_density_dip function with mixture of Gaussians',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--n1', type=int, default=1000,
                        help='Number of samples from first Gaussian')
    parser.add_argument('--n2', type=int, default=1000,
                        help='Number of samples from second Gaussian')
    parser.add_argument('--mu1', type=float, default=0.0,
                        help='Mean of first Gaussian')
    parser.add_argument('--mu2', type=float, default=5.0,
                        help='Mean of second Gaussian')
    parser.add_argument('--sigma1', type=float, default=1.0,
                        help='Standard deviation of first Gaussian')
    parser.add_argument('--sigma2', type=float, default=1.0,
                        help='Standard deviation of second Gaussian')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of trials to run')
    
    return parser.parse_args()

def main():
    """Run the benchmark with parameters from command line or defaults."""
    # Clear results directory
    results_dir = 'benchmarks/results'
    if os.path.exists(results_dir):
        print(f"Clearing results directory: {results_dir}")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    print()
    
    args = parse_args()
    
    results = run_benchmark(
        n1=args.n1,
        n2=args.n2,
        mu1=args.mu1,
        mu2=args.mu2,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        n_trials=args.n_trials
    )
    
    print("="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
