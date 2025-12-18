# Gaussian Mixture Model (GMM) Source Approximation

This directory contains code for fitting a Gaussian Mixture Model to CIFAR-10 training data to create a flexible source distribution for flow matching models.

## Overview

**Motivation**: Standard Gaussian noise `x₀ ~ N(0, I)` is a simple but potentially suboptimal source distribution. Real image data often has multi-modal structure that can be better captured by a mixture of Gaussians. By fitting a GMM to the training data, we create a more informed source distribution that better approximates the data manifold.

**Key Idea**:
1. Flatten all CIFAR-10 training images to vectors (32×32×3 = 3072 dimensions)
2. Optionally apply PCA for dimensionality reduction
3. Fit a Gaussian Mixture Model with K components
4. Use the fitted GMM as the source distribution for flow matching

## Files

- **`gmm_fit.py`**: Main script to fit GMM to CIFAR-10 and compute normalization statistics

## Generated Files

The script generates two files in `../../data/`:

### 1. GMM Model File
**Format**: `cifar10_gmm{K}_pca{D}_{covariance_type}.pkl`

Contains:
- **`gmm`**: Fitted sklearn GaussianMixture model with K components
- **`pca`**: Fitted PCA model (if dimensionality reduction used), or None

**Example**: `cifar10_gmm2_pca3072_full.pkl`
- 2 Gaussian components
- No PCA (3072 = full dimensionality)
- Full covariance matrices

### 2. Normalization Statistics
**Format**: `cifar10_gmm{K}_pca{D}_{covariance_type}_stats.pt`

Contains:
- **`mean`**: Mean of GMM samples (for normalization)
- **`std`**: Standard deviation of GMM samples (for normalization)

These statistics ensure that samples from the GMM have zero mean and unit variance, similar to standard Gaussian noise.

## How to Run

### Prerequisites

```bash
pip install torch torchvision scikit-learn numpy joblib tqdm
```

### Basic Usage

Fit a GMM with default parameters (2 components, no PCA, full covariance):

```bash
cd source_approx/GMM
python gmm_fit.py
```

### Advanced Usage

```bash
python gmm_fit.py \
    --output_dir ../../data \
    --n_components 10 \
    --n_pca 300 \
    --covariance_type full \
    --n_stat_samples 50000 \
    --batch_size 128
```

### Command Line Options

- `--output_dir`: Where to save output files (default: `../../data`)
- `--data_root`: Root directory for CIFAR-10 data (default: `./data`)
- `--n_components`: Number of Gaussian components in the mixture (default: 2)
- `--n_pca`: Number of PCA components; use 3072 to skip PCA (default: 3072)
- `--covariance_type`: Type of covariance matrix (default: `full`)
  - `full`: Each component has its own full covariance matrix
  - `tied`: All components share the same covariance matrix
  - `diag`: Diagonal covariance matrices (faster, less expressive)
  - `spherical`: Single variance per component (fastest, least expressive)
- `--n_stat_samples`: Number of samples for computing normalization statistics (default: 50000)
- `--batch_size`: Batch size for sampling (default: 128)
- `--random_state`: Random seed for reproducibility (default: 0)
- `--augmentation`: Use random horizontal flip augmentation

### Example Configurations

**Simple 2-component GMM (fast)**:
```bash
python gmm_fit.py --n_components 2 --covariance_type full
```

**10-component GMM with PCA compression**:
```bash
python gmm_fit.py --n_components 10 --n_pca 300 --covariance_type full
```

**Large 100-component GMM (slow, high memory)**:
```bash
python gmm_fit.py --n_components 100 --covariance_type full
```

## Expected Runtime

- **2 components, no PCA**: ~30 seconds
- **10 components, no PCA**: ~2-5 minutes
- **100 components, no PCA**: ~15-30 minutes

**Note**: Full covariance with high-dimensional data (3072D) requires significant memory. Consider using PCA or `covariance_type='tied'` for large K.

## How It Works

### 1. Data Preparation

Load all 50,000 CIFAR-10 training images and flatten to vectors:
```
(50000, 3, 32, 32) → (50000, 3072)
```

Normalize to [-1, 1] range.

### 2. Optional PCA

If `n_pca < 3072`, reduce dimensionality:
```
(50000, 3072) → (50000, n_pca)
```

This makes GMM fitting faster and reduces memory usage, but loses some information.

### 3. GMM Fitting

Fit a mixture of K Gaussians using EM algorithm:
```
p(x) = Σᵢ wᵢ · N(x | μᵢ, Σᵢ)
```

Where:
- `wᵢ`: Mixture weights (sum to 1)
- `μᵢ`: Mean of component i
- `Σᵢ`: Covariance matrix of component i

### 4. Normalization Statistics

Sample 50,000 points from the fitted GMM and compute mean/std:
```python
x ~ GMM(w, μ, Σ)
mean = E[x]
std = sqrt(Var[x])
```

These are used to normalize samples during training:
```python
x_normalized = (x - mean) / std
```

## Usage in Flow Matching

To use the GMM source in your flow matching model:

```python
import joblib
import torch
from utils.gmm import GMM_TORCH

# Load the fitted model
model_data = joblib.load('data/cifar10_gmm2_pca3072_full.pkl')
gmm_sklearn = model_data['gmm']

# Load normalization statistics
stats = torch.load('data/cifar10_gmm2_pca3072_full_stats.pt')

# Create PyTorch GMM sampler
gmm = GMM_TORCH(
    gmm=gmm_sklearn,
    gmm_stats=stats,
    device='cuda'
)

# Sample from GMM
x0 = gmm.sample(batch_size)  # Returns (B, 3, 32, 32) normalized tensor
```

## Interpretation

- **K=1**: Equivalent to a single Gaussian (similar to standard noise, but with learned covariance)
- **K=2-10**: Captures major modes in the data (e.g., different image types)

The GMM provides a data-dependent source distribution that can potentially improve flow matching training by starting from a distribution closer to the target data.

## References

- See paper page 11 for motivation and theoretical background
- Gaussian Mixture Models are a classical method in machine learning for density estimation
- In flow matching, they provide an alternative to standard Gaussian noise as the source distribution