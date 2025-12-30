import numpy as np
from scipy.stats import chi2


def poisson_ci_lower(k, conf):
    """
    Exact lower bound of a (1 - alpha) confidence interval for lambda
    of a Poisson(k). Uses chi-square distribution.
    
    Parameters:
        k : int  (observed count)
        conf : float (confidence level, e.g. 0.95 or 0.50)
    
    Returns:
        float : Lower bound of the confidence interval
    """
    alpha = 1 - conf
    if k == 0:
        return 0.0
    return 0.5 * chi2.ppf(alpha / 2, 2 * k)


def poisson_ci_upper(k, conf):
    """
    Exact upper bound of a (1 - alpha) confidence interval for lambda
    of a Poisson(k). Uses chi-square distribution.
    
    Parameters:
        k : int  (observed count)
        conf : float (confidence level, e.g. 0.95 or 0.50)
    
    Returns:
        float : Upper bound of the confidence interval
    """
    alpha = 1 - conf
    return 0.5 * chi2.ppf(1 - alpha / 2, 2 * k + 2)


def estimate_density_dip(
    data: np.ndarray,
    a: float,
    b: float,
    conf: float = 0.5
) -> dict:
    """
    Estimate the density dip between two cluster centroids.
    
    This function evaluates multiple histogram bin sizes to find the minimum
    density point between centroids a and b, using Poisson confidence intervals
    to account for statistical uncertainty in the bin counts.
    
    Parameters:
        data : np.ndarray
            1D array of data points
        a : float
            First centroid location (left)
        b : float
            Second centroid location (right)
        conf : float, optional
            Confidence level for Poisson intervals (default=0.5)
    
    Returns:
        dict : Dictionary containing:
            - 'c': estimated cutpoint (midpoint of minimum density bin)
            - 'density_a': density estimate at left endpoint
            - 'density_b': density estimate at right endpoint
            - 'density_c': density estimate at cutpoint
    """
    # Ensure a <= b
    if a > b:
        a, b = b, a
    
    # Number of bins to try
    num_bins_options = [10, 20, 40, 80, 160]
    
    best_dip = -np.inf  # We want the largest dip
    best_result = None
    
    # Try each binning option
    for num_bins in num_bins_options:
        # Calculate bin width such that we have the desired number of bins
        # The first edge is at a-h/2 and the last edge is at b+h/2
        # With num_bins bins, we have: a - h/2 + num_bins*h = b + h/2
        # Solving: h(num_bins - 1) = b - a
        h = (b - a) / (num_bins - 1) if num_bins > 1 else (b - a)
        
        # Create bin edges: [a-h/2, a+h/2, a+3h/2, ..., b+h/2]
        # This creates bins centered at a, a+h, a+2h, ..., b
        bin_edges = np.array([a - h/2 + i * h for i in range(num_bins + 1)])
        
        # Compute histogram counts
        # Note: We don't filter data because edge bins extend beyond [a, b]
        counts, _ = np.histogram(data, bins=bin_edges)
        
        # Skip if we have no data in bins
        if len(counts) == 0 or np.sum(counts) == 0:
            continue
        
        # Calculate lower and upper confidence bounds for each bin count
        counts_lower = np.array([poisson_ci_lower(k, conf) for k in counts])
        counts_upper = np.array([poisson_ci_upper(k, conf) for k in counts])

        # Find the bin with minimum count (this gives us the cutpoint)
        min_idx = np.argmin(counts)
        
        # The cutpoint c is the midpoint of the minimum bin
        c = a + min_idx * h
        
        # Calculate density dip score
        # Dip = min(endpoint densities) / max(density upper bounds)
        # All densities are normalized by bin width h
        density_left_lower = counts_lower[0] / h
        density_right_lower = counts_lower[-1] / h
        density_c_upper = counts_upper[min_idx] / h
        
        # Calculate dip score
        if density_c_upper > 0:
            dip_score = min(density_left_lower, density_right_lower) / density_c_upper
        else:
            dip_score = 0.0
        
        # Keep track of the best (largest) dip
        if dip_score > best_dip:
            best_dip = dip_score
            best_result = {
                'c': c,
                'density_a': density_left_lower,
                'density_b': density_right_lower,
                'density_c': density_c_upper,
                'num_bins': num_bins,  # For debugging
                'bin_width': h  # For debugging
            }
    
    # If no valid result was found, return default values
    if best_result is None:
        return {
            'c': (a + b) / 2,
            'density_a': 0.0,
            'density_b': 0.0,
            'density_c': 0.0
        }

    return best_result
