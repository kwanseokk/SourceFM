# PCA-Based Pruned Sampling

This directory contains the implementation of our proposed **Pruned Sampling** method - the key contribution that enables post-hoc performance improvements without retraining.

## Overview

**Key Idea**: Train with full Gaussian (robust omnidirectional learning) → Sample from pruned regions at inference (avoid data-sparse, poorly-learned areas).

**Why It Works**:
- Full Gaussian training ensures all flow directions receive sufficient training signal
- PCA identifies principal directions in the data manifold
- Rejection sampling eliminates initializations in data-sparse regions
- Can be applied **post-hoc** to any pre-trained flow matching model!

---

## File

- **`find_pruning_axis_with_pca.py`**: Main script for computing PCA-based pruned axes

---

## Quick Start (CIFAR-10)

### Option 1: Download Pre-computed Files (Recommended)

Download pre-computed pruned axes from Google Drive:
- [https://drive.google.com/drive/folders/19Ayyfoeddp2yR3cDXJELxYEZJ7xfV8mX?usp=sharing](https://drive.google.com/drive/folders/19Ayyfoeddp2yR3cDXJELxYEZJ7xfV8mX?usp=sharing)

Place `pca_pruned_flip_under_1e-2.pt` in the `data/` folder.

### Option 2: Compute Pruned Axes from Scratch

```bash
cd 2_pruning

# Standard threshold (0.01)
python find_pruning_axis_with_pca.py \
    --dataset cifar10 \
    --rejection_threshold 0.01 \
    --pca_method incremental

# Custom threshold
python find_pruning_axis_with_pca.py \
    --dataset cifar10 \
    --rejection_threshold 0.02 \
    --pca_method incremental
```

**Output**: `../data/pca_pruned_flip_under_1e-2.pt`

This file contains the PCA axes that should be **accepted** (not rejected) during sampling.

---

## How It Works

### 1. Load and Normalize Data
- Loads CIFAR-10 training set (50K images)
- Applies horizontal flip augmentation → 100K samples
- Flattens to vectors (3072-D) and L2-normalizes to unit sphere

### 2. Compute PCA Components
- Fits PCA on normalized data
- Creates bidirectional centroids: both +PCA and -PCA directions
- Result: 2×3072 = 6144 candidate axes

### 3. Find Data-Sparse Directions
For each axis:
- Compute max cosine similarity to all data points
- If `max_similarity ≤ threshold` → Mark as **rejected** (data-sparse)
- Otherwise → Mark as **accepted** (data-rich)

### 4. Save Pruned Axes
- Saves only the **accepted** axes (where data exists)
- These axes define safe sampling regions

---

## Usage in Flow Matching

The pruned axes are used during **inference** in `compute_fid_cifar.py`:

```bash
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm \
    --source pca_pruned \
    --threshold 0.048 \
    --start_step 80000 --end_step 180000 --step 10000 \
    --num_gen 50000
```

During sampling:
1. Sample initial noise `x₀ ~ N(0, I)`
2. Check if `x₀` is close to any pruned (rejected) axis
3. If yes → **reject** and resample
4. If no → **accept** and run flow matching

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar10` | Dataset to use (`cifar10`, `imagenet64`, `imagenet256`, `sdv2_imagenet32`, `ffhq`) |
| `--rejection_threshold` | `0.01` | Cosine similarity threshold for pruning |
| `--pca_method` | `incremental` | PCA computation method |
| `--cache_path_root` | `../data` | Directory for cache files |
| `--device` | `cuda` | Device to use (`cuda` or `cpu`) |
| `--pca_batch_size` | `None` | Batch size for incremental PCA (auto if None) |
| `--test_equivalence` | `False` | Test mathematical equivalence of PCA methods |

### PCA Methods

| Method | When to Use | Memory | Speed |
|--------|-------------|--------|-------|
| `incremental` | **Recommended** for most cases | Low | Medium |
| `randomized` | Very large datasets (>1M samples) | Medium | Fast |
| `fast` | Small datasets that fit in GPU memory | High | Very Fast |
| `batched` | Original method, fallback option | High | Slow |

---

## Other Datasets

### ImageNet64
```bash
python find_pruning_axis_with_pca.py \
    --dataset imagenet64 \
    --rejection_threshold 0.01 \
    --pca_method randomized
```
**Output**: `../data/imagenet64_pca_pruned_flip_under_1e-2.pt`

### ImageNet256
```bash
python find_pruning_axis_with_pca.py \
    --dataset imagenet256 \
    --rejection_threshold 0.01 \
    --pca_method incremental
```
**Output**: `../data/imagenet256_pca_pruned_flip_under_1e-2.pt`

### Stable Diffusion v2 Latents (ImageNet32)
```bash
python find_pruning_axis_with_pca.py \
    --dataset sdv2_imagenet32 \
    --rejection_threshold 0.01 \
    --pca_method incremental
```
**Output**: `../data/sdv2_imagenet32_pca_pruned_flip_under_1e-2.pt`

### FFHQ-256
```bash
python find_pruning_axis_with_pca.py \
    --dataset ffhq \
    --rejection_threshold 0.01 \
    --pca_method incremental
```
**Output**: `../data/ffhq_pca_pruned_flip_under_1e-2.pt`

---

## Threshold Selection

The rejection threshold determines how aggressively to prune:

| Threshold | Pruning | Use Case |
|-----------|--------------|----------|
| `0.005` | High | Very conservative, removes only extremely sparse regions |
| `0.01` | Medium | **Recommended** - good balance |
| `0.02` | Low | More aggressive, may remove some valid regions |

**During inference**, a separate threshold (e.g., `0.048` for CIFAR-10) is used for rejection sampling.

---

## Caching

The script uses caching to speed up repeated runs:

- **PCA components**: `../data/{dataset}/pca_components_{method}.pt`
- **Max similarities**: `../data/{dataset}/max_similarities.pt`

Delete these files to force recomputation.

---

## Memory Considerations

For large datasets, use `incremental` or `randomized` PCA:

```bash
# For datasets that don't fit in memory
python find_pruning_axis_with_pca.py \
    --dataset imagenet256 \
    --pca_method incremental \
    --pca_batch_size 1000
```

The script automatically estimates memory usage and provides warnings.

---

## Results

**CIFAR-10 (threshold=0.01 for pruning, 0.048 for inference):**

| Method | NFE=5 | NFE=10 | NFE=20 | NFE=100 |
|--------|-------|--------|--------|---------|
| OT-CFM Baseline | 19.80 | 10.95 | 7.41 | 4.40 |
| + Pruned | **17.24** | **9.17** | **6.34** | **4.15** |
| I-CFM Baseline | 34.49 | 13.30 | 7.75 | 4.35 |
| + Pruned | **28.78** | **10.37** | **6.40** | **3.97** |

**ImageNet64 (threshold=0.01):**

| Method | NFE=100 |
|--------|---------|
| OT-CFM Baseline | 9.10 |
| + Pruned | **8.78** |

---

## Notes

- **Data augmentation**: CIFAR-10 uses horizontal flip augmentation (100K samples)
- **L2 normalization**: All data is normalized to unit sphere before PCA
- **Bidirectional axes**: Both +/- PCA directions are considered
- **Post-hoc application**: Works on any pre-trained model without retraining!
- **Computational cost**: One-time preprocessing, negligible inference overhead
