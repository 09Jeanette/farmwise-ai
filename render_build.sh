#!/bin/bash

# Render deployment build script with memory optimization

echo "Starting optimized deployment..."

# Set environment variables for memory optimization
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Install dependencies with memory constraints
pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Clear pip cache to save memory
pip cache purge

echo "Build completed with memory optimizations"
