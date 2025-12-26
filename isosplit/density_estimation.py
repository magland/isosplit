"""
1D Density Estimation for IsoSplit Clustering

This module provides various methods for estimating density in 1D data,
used to determine separation between clusters in the IsoSplit algorithm.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d


@dataclass
class DensityEstimationConfig:
    """
    Configuration for 1D density estimation in IsoSplit.
    
    Parameters
    ----------
    method : {'gmm', 'kde', 'histogram'}
        Density estimation method to use:
        - 'histogram': Histogram-based estimation (default, fastest)
        - 'gmm': Gaussian Mixture Model (slowest)
        - 'kde': Kernel Density Estimation (fast)
    
    gmm_max_components : int, default=5
        Maximum number of Gaussian components to consider for GMM
    gmm_n_components : int or 'auto', default='auto'
        Number of components for GMM. If 'auto', uses BIC to select optimal number
    gmm_random_state : int, default=42
        Random state for GMM reproducibility
    
    kde_bandwidth : float or None, default=None
        Bandwidth for KDE. If None, uses Scott's rule
    kde_bw_method : {'scott', 'silverman'}, default='scott'
        Method for automatic bandwidth selection in KDE
    
    hist_n_bins : int or 'auto', default='auto'
        Number of bins for histogram method. If 'auto', uses Freedman-Diaconis rule
        to automatically determine optimal bin count based on data distribution.
    hist_smoothing : bool, default=True
        Whether to apply Gaussian smoothing to histogram
    hist_smoothing_sigma : float, default=0.1
        Standard deviation for Gaussian smoothing kernel as a fraction of data standard deviation.
        This makes smoothing scale-independent. Default 0.1 means smoothing width is 10% of data std.
    """
    method: Literal['histogram', 'gmm', 'kde'] = 'histogram'
    
    # GMM-specific parameters
    gmm_max_components: int = 5
    gmm_n_components: str | int = 'auto'
    gmm_random_state: int = 42
    
    # KDE-specific parameters
    kde_bandwidth: Optional[float] = None
    kde_bw_method: Literal['scott', 'silverman'] = 'scott'
    
    # Histogram-specific parameters
    hist_n_bins: int | str = 'auto'
    hist_smoothing: bool = True
    hist_smoothing_sigma: float = 0.1


def estimate_density_dip(
    data, a, b, config: DensityEstimationConfig
):
    """
    Estimate density at points a and b, and find the minimum density point c between them.
    
    Dispatches to the appropriate method based on configuration.

    Parameters:
    -----------
    data : array-like
        Collection of data points (floats) from which to estimate the density
    a : float
        First point (must be < b)
    b : float
        Second point (must be > a)
    config : DensityEstimationConfig
        Configuration specifying which method to use and its parameters

    Returns:
    --------
    dict with keys:
        'density_a': float - estimated density at point a
        'density_b': float - estimated density at point b
        'c': float - point of minimum density between a and b
        'density_c': float - estimated density at point c

    Raises:
    -------
    ValueError: If a >= b or if data is empty
    """
    # Input validation
    data = np.array(data).ravel()

    if len(data) == 0:
        raise ValueError("Data cannot be empty")

    if a >= b:
        raise ValueError(f"Point a ({a}) must be less than point b ({b})")

    # Dispatch to appropriate method
    if config.method == 'gmm':
        return _estimate_density_dip_gmm(data, a, b, config)
    elif config.method == 'kde':
        return _estimate_density_dip_kde(data, a, b, config)
    elif config.method == 'histogram':
        return _estimate_density_dip_histogram(data, a, b, config)
    else:
        raise ValueError(f"Unknown density estimation method: {config.method}")


def _estimate_density_dip_gmm(
    data: np.ndarray, a: float, b: float, config: DensityEstimationConfig
):
    """
    Estimate density dip using Gaussian Mixture Modeling.
    
    This is the original method - slower but flexible.
    """
    data = data.reshape(-1, 1)
    
    n_components = config.gmm_n_components
    max_components = config.gmm_max_components
    random_state = config.gmm_random_state

    # Determine optimal number of components if 'auto'
    if n_components == "auto":
        best_bic = np.inf
        best_n = 1

        for n in range(1, min(max_components + 1, len(data))):
            gmm_temp = GaussianMixture(n_components=n, random_state=random_state)
            gmm_temp.fit(data)
            bic = gmm_temp.bic(data)

            if bic < best_bic:
                best_bic = bic
                best_n = n

        n_components = best_n

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)

    # Function to compute density at a point
    def density_at_point(x):
        x_array = np.array([[x]])
        return np.exp(gmm.score_samples(x_array)[0])

    # Estimate densities at a and b
    density_a = density_at_point(a)
    density_b = density_at_point(b)

    # Find point of minimum density between a and b
    result = minimize_scalar(
        lambda x: density_at_point(x),
        bounds=(a, b),
        method="bounded",
    )

    c = result.x
    density_c = density_at_point(c)

    return {
        "density_a": density_a,
        "density_b": density_b,
        "c": c,
        "density_c": density_c,
    }


def _estimate_density_dip_kde(
    data: np.ndarray, a: float, b: float, config: DensityEstimationConfig
):
    """
    Estimate density dip using Kernel Density Estimation.
    
    Much faster than GMM, good balance of speed and accuracy.
    """
    # Create KDE
    if config.kde_bandwidth is not None:
        # Use specified bandwidth (need to override bw_method)
        kde = gaussian_kde(data, bw_method=config.kde_bandwidth / data.std(ddof=1))
    else:
        # Use automatic bandwidth selection
        kde = gaussian_kde(data, bw_method=config.kde_bw_method)

    # Estimate densities at a and b
    density_a = kde(a)[0]
    density_b = kde(b)[0]

    # Find point of minimum density between a and b
    result = minimize_scalar(
        lambda x: kde(x)[0],
        bounds=(a, b),
        method="bounded",
    )

    c = result.x
    density_c = kde(c)[0]

    return {
        "density_a": density_a,
        "density_b": density_b,
        "c": c,
        "density_c": density_c,
    }


def _estimate_density_dip_histogram(
    data: np.ndarray, a: float, b: float, config: DensityEstimationConfig
):
    """
    Estimate density dip using histogram-based method.
    
    Fastest method, good for large datasets where speed is critical.
    """
    # Determine number of bins
    if config.hist_n_bins == 'auto':
        # Use Freedman-Diaconis rule for automatic bin selection
        # bins = 'fd' uses the Freedman-Diaconis rule in numpy
        bins = 'fd'
    else:
        bins = config.hist_n_bins
    
    # Create histogram
    hist, edges = np.histogram(data, bins=bins, density=True)
    
    # Get actual number of bins (needed for smoothing calculation)
    n_bins = len(hist)
    
    # Apply smoothing if requested
    if config.hist_smoothing:
        # Convert sigma from data units to bin units for scale-independence
        # sigma in data units = config.hist_smoothing_sigma * data.std()
        # bin width = (data range) / n_bins
        bin_width = (edges[-1] - edges[0]) / n_bins
        sigma_data_units = config.hist_smoothing_sigma * data.std(ddof=1)
        sigma_bins = sigma_data_units / bin_width
        hist = gaussian_filter1d(hist, sigma=sigma_bins)
    
    # Bin centers
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Interpolate density at a and b
    density_a = np.interp(a, centers, hist)
    density_b = np.interp(b, centers, hist)
    
    # Find minimum density point between a and b
    # Restrict to bins between a and b
    mask = (centers >= a) & (centers <= b)
    if not np.any(mask):
        # If no bins between a and b, use midpoint
        c = (a + b) / 2
        density_c = np.interp(c, centers, hist)
    else:
        centers_range = centers[mask]
        hist_range = hist[mask]
        min_idx = np.argmin(hist_range)
        c = centers_range[min_idx]
        density_c = hist_range[min_idx]
    
    return {
        "density_a": density_a,
        "density_b": density_b,
        "c": c,
        "density_c": density_c,
    }
