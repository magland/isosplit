#!/bin/bash

# Benchmark script for estimate_density_dip
# Modify the parameters below to test different configurations

python benchmarks/density_dip_benchmark.py \
    --n1 5000 \
    --n2 2000 \
    --mu1 0.0 \
    --mu2 25.0 \
    --sigma1 10.0 \
    --sigma2 5.0 \
    --n-trials 20
