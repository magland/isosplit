# IsoSplit Performance Profiling

This directory contains scripts and tools for profiling the IsoSplit clustering algorithm to identify performance bottlenecks.

## Overview

The profiling suite includes several tools to analyze different aspects of performance:

1. **Function-level profiling** - Identify which functions consume the most time
2. **Line-level profiling** - Find bottlenecks within specific functions
3. **Memory profiling** - Track memory usage and identify memory bottlenecks
4. **Scaling analysis** - Understand how performance scales with data size

## Scripts

### 1. `profile_isosplit.py` - Main Profiling Script

Comprehensive profiling using Python's built-in `cProfile` module.

**Usage:**
```bash
cd /var/home/magland/src/isosplit
python profile/profile_isosplit.py
```

**Features:**
- Tests multiple dataset sizes (small, medium, large)
- Generates detailed profiling reports saved to `profile_results_*.txt`
- Identifies top time-consuming functions
- Shows cumulative time spent in each function

**Output:**
- Console output with summary statistics
- `profile_results_1.txt` - Small dataset results
- `profile_results_2.txt` - Medium dataset results
- `profile_results_3.txt` - Large dataset results

### 2. `line_profiler_analysis.py` - Line-by-Line Profiling

Fine-grained profiling at the line level using `line_profiler`.

**Setup:**
```bash
pip install line_profiler
```

**Usage:**
```bash
# To profile specific functions, add @profile decorator to them in isosplit/core.py
kernprof -l -v profile/line_profiler_analysis.py
```

**Features:**
- Shows time spent on each line of code
- Identifies exact bottleneck lines within functions
- Useful for optimizing specific function implementations

### 3. `memory_profiler_analysis.py` - Memory Usage Analysis

Tracks memory consumption during execution.

**Setup:**
```bash
pip install memory_profiler
```

**Usage:**
```bash
python -m memory_profiler profile/memory_profiler_analysis.py
```

**Features:**
- Line-by-line memory usage tracking
- Identifies memory-intensive operations
- Helps optimize memory footprint

### 4. `benchmark_scaling.py` - Scaling Behavior Analysis

Tests how performance scales with different parameters.

**Usage:**
```bash
python profile/benchmark_scaling.py
```

**Features:**
- Tests scaling with number of samples
- Tests scaling with number of dimensions
- Tests scaling with number of clusters
- Generates visualization plots

**Output:**
- `scaling_samples.png` - Sample scaling plot
- `scaling_dimensions.png` - Dimension scaling plot
- `scaling_clusters.png` - Cluster scaling plot
- `scaling_summary.png` - Combined summary plot

## Quick Start

Run the main profiling script to get started:

```bash
cd /var/home/magland/src/isosplit
python profile/profile_isosplit.py
```

This will:
1. Generate test data with multiple clusters
2. Run profiling on different dataset sizes
3. Create detailed reports in `profile_results_*.txt`
4. Print summary statistics to console

## Expected Bottlenecks

Based on the algorithm implementation, likely bottlenecks include:

### Computational Bottlenecks:
1. **LDA projection** (`_merge_test`) - O(n*d²) for n samples, d dimensions
2. **Gaussian Mixture Modeling** (`estimate_density_dip`) - Iterative optimization
3. **Distance matrix computation** (`_compute_distance_matrix`) - O(k²*d) for k clusters
4. **Cluster centroid updates** - Called repeatedly during merging

### Algorithmic Bottlenecks:
1. **Iterative merging loop** - Number of iterations depends on initial_k
2. **Pairwise cluster comparisons** - O(k²) comparisons in worst case
3. **Point redistribution** - Reassigning points between clusters

### Memory Bottlenecks:
1. **Distance matrix storage** - O(k²) space for k clusters
2. **Data copies during merge testing** - Creating cluster subsets
3. **Centroid array storage** - O(k*d) space

## Optimization Strategies

After running profiling, consider these optimization approaches:

1. **Vectorization** - Replace loops with NumPy operations
2. **Caching** - Store frequently computed values (e.g., centroids)
3. **Early termination** - Skip unnecessary comparisons
4. **Memory efficiency** - Use in-place operations where possible
5. **Algorithmic improvements** - Better data structures or algorithms

## Understanding the Output

### cProfile Output Fields:
- **ncalls** - Number of calls to the function
- **tottime** - Total time spent in the function (excluding subcalls)
- **percall** - tottime / ncalls
- **cumtime** - Cumulative time (including subcalls)
- **percall** - cumtime / ncalls

### Focus Areas:
- Functions with high **cumtime** are the main bottlenecks
- Functions with high **ncalls** might benefit from caching
- Functions with high **tottime** need internal optimization

## Results Storage

All profiling results are stored in the `profile/` directory:
- `*.txt` - Text-based profiling reports
- `*.png` - Visualization plots
- `*.lprof` - Line profiler binary data (if generated)

## Next Steps

1. Run `profile_isosplit.py` to identify main bottlenecks
2. Use `line_profiler_analysis.py` on specific bottleneck functions
3. Check memory usage with `memory_profiler_analysis.py`
4. Analyze scaling behavior with `benchmark_scaling.py`
5. Implement optimizations based on findings
6. Re-run profiling to measure improvements

## Notes

- Profiling adds overhead - actual performance will be faster without profiling
- Use consistent test data for comparing before/after optimization
- Focus on optimizing the most significant bottlenecks first (80/20 rule)
- Consider algorithmic improvements before micro-optimizations
