# Directional Alignment with Spherical K-Means

Spherical k-means clustering on CIFAR-10 for creating directionally-aligned source distributions.

## Overview

Instead of using isotropic Gaussian noise, we cluster normalized CIFAR-10 images on the unit hypersphere and fit von Mises-Fisher (vMF) distributions to each cluster. This provides structured, directional source distributions for flow matching.

## Files

- **`fit_spherical_kmeans.py`**: Main script for clustering and vMF fitting
- **`lab.ipynb`**: Experimental notebook

## Generated Files

### spherical_kmeans{K}_flip.pt

```python
{
    'centroid':      Tensor(K, 3072)       # Cluster centroids (unit vectors)
    'assignments':   Tensor(N,)            # Cluster assignments
    'kappa':         Tensor(K,)            # vMF concentration parameters
    'emp_angles':    Tensor(K, 100)        # Quantile-compressed angle distributions
    'theta_min':     Tensor(K,)            # Min angle per cluster
    'theta_max':     Tensor(K,)            # Max angle per cluster
    'sample_ratio':  Tensor(K,)            # Cluster proportions
    'norm_mu':       Tensor(K, 1, 3, 1, 1) # Per-cluster channel means
    'norm_std':      Tensor(K, 1, 3, 1, 1) # Per-cluster channel stds
    'all_norm_std':  Tensor([])            # Global std (scalar)
}
```

### clusters_spherical_kmeans{K}_flip.pt

Contains only the cluster assignments tensor.

## Usage

### Create spherical_kmeans3_flip.pt

```bash
python fit_spherical_kmeans.py --n_clusters 3 --augment_flip
```

### Elbow Study

Find optimal K using silhouette scores:

```bash
python fit_spherical_kmeans.py \
    --elbow_study \
    --elbow_min 2 \
    --elbow_max 10 \
    --augment_flip
```

Generates `elbow_study.png` with silhouette scores and inertia curves.

### Options

```bash
--n_clusters 3              # Number of clusters
--augment_flip              # Include horizontal flip (doubles data to 100k)
--data_root ../data         # CIFAR-10 data directory
--output_dir ../data        # Output directory
--elbow_study               # Run elbow study before clustering
--elbow_min 2               # Min K for elbow study
--elbow_max 10              # Max K for elbow study
--max_iter 500              # Max k-means iterations
--kappa_method newton       # Kappa estimation: banerjee, sra, newton
--device cuda               # Device: cuda or cpu
```

## How It Works

1. **Preprocess**: Load CIFAR-10, normalize to [-1,1], flatten to (N, 3072), L2 normalize to unit sphere
2. **K-means++**: Initialize centroids with k-means++ (better than random)
3. **Spherical k-means**: Cluster using cosine similarity, converge on unit sphere
4. **vMF fitting**: Estimate μ (centroid) and κ (concentration) per cluster
5. **Statistics**: Compute angle distributions, channel means/stds per cluster

**With flip augmentation**: Horizontally flip all images → 100k samples instead of 50k.

## Integration

```python
import torch

# Load clustering
ckpt = torch.load('data/spherical_kmeans3_flip.pt')
centroids = ckpt['centroid']      # (3, 3072)
kappa = ckpt['kappa']             # (3,)
sample_ratio = ckpt['sample_ratio']  # (3,)

# Sample cluster
k = torch.multinomial(sample_ratio, 1).item()

# Sample from vMF(μ_k, κ_k)
# ... (use vMF sampler)
```

## References

- Silhouette score: Rousseeuw (1987)
