# DCT-Based Source Approximation

This directory contains code for creating and using DCT (Discrete Cosine Transform) based source distributions for flow matching models. Instead of using standard Gaussian noise as the source distribution, we use DCT-masked noise that better approximates the frequency characteristics of natural images.

## Overview

**Motivation**: Natural images have characteristic frequency distributions when transformed to the DCT domain (similar to JPEG compression). Certain DCT frequency components have very low variance across images, meaning they contribute little information. By masking (zeroing out) these low-variance components in the source distribution, we create a more informed prior that's closer to the structure of natural images.

**Key Idea**:
1. Analyze CIFAR-10 training images in the DCT domain
2. Identify frequency components with low mean and variance
3. Create masks to zero out these components
4. Use the masked DCT transform as a source distribution for flow matching

## Files

### Core Scripts

- **`create_dct_masks.py`**: Main script to generate all DCT masks and statistics from CIFAR-10
- **`compute_fid_dctmask_torch.py`**: Script to evaluate FID scores of models trained with DCT-masked source

### Notebooks

- **`visulize.ipynb`**: Visualization notebook showing:
  - DCT mean and standard deviation heatmaps for each channel
  - Sampling from DCT-based distributions

- **`dct.ipynb`**: Experimental notebook exploring:
  - JPEG compression mechanics
  - DCT coefficient analysis
  - Mask creation experiments

## Generated Files

The `create_dct_masks.py` script generates the following files in `../../data/`:

### 1. DCT Statistics (Mean and Variance)
These files contain the mean and variance of DCT coefficients computed across all CIFAR-10 training images:

- `luminance_mean.npy`, `luminance_var.npy` (8×8 arrays)
- `chrominance_cb_mean.npy`, `chrominance_cb_var.npy` (8×8 arrays)
- `chrominance_cr_mean.npy`, `chrominance_cr_var.npy` (8×8 arrays)

Each position in the 8×8 array corresponds to a DCT frequency component.

### 2. DCT Masks
Boolean masks indicating which DCT coefficients to zero out:

**Weak Masks** (more conservative, fewer masked components):
- `luminance_mask_weak.npy`
- `chrominance_cb_mask_weak.npy`
- `chrominance_cr_mask_weak.npy`

**Strong Masks** (more aggressive, more masked components):
- `luminance_mask_strong.npy`
- `chrominance_cb_mask_strong.npy`
- `chrominance_cr_mask_strong.npy`

**Masking Criteria**:
- **Weak**: `|mean| < 0.1` AND `variance < 2.0`
- **Strong**: `|mean| < 0.1` AND `variance < 4.0`

### 3. Noise Normalization Statistics
These files contain normalization statistics for Gaussian noise after DCT masking:

- `noise_dctmask6_weak_torch.pt`: Contains `{'mean': Tensor, 'std': Tensor}` for weak masking
- `noise_dctmask6_strong_torch.pt`: Contains `{'mean': Tensor, 'std': Tensor}` for strong masking

**Purpose**: When using DCT masking as a source distribution, we need to normalize the masked noise. These statistics are computed by:
1. Generating 1M Gaussian noise samples
2. Processing them through DCT masking
3. Computing the mean and std of the resulting distribution

## How to Run

### Prerequisites

```bash
pip install torch torchvision numpy opencv-python scipy tqdm
```

### Generate All Masks and Statistics

```bash
cd source_approx/DCT
python create_dct_masks.py
```

### Command Line Options

```bash
python create_dct_masks.py \
    --output_dir ../../data \
    --mean_threshold 0.1 \
    --var_threshold_weak 2.0 \
    --var_threshold_strong 4.0 \
    --num_noise_samples 1000000 \
    --batch_size 512
```

**Arguments**:
- `--output_dir`: Where to save all generated files (default: `../../data`)
- `--mean_threshold`: Absolute mean threshold for masking (default: 0.1)
- `--var_threshold_weak`: Variance threshold for weak masks (default: 2.0)
- `--var_threshold_strong`: Variance threshold for strong masks (default: 4.0)
- `--num_noise_samples`: Number of noise samples for statistics (default: 1,000,000)
- `--batch_size`: Batch size for processing (default: 500)

### Expected Runtime

- **DCT statistics computation**: ~5-10 minutes (50,000 images × 16 patches)
- **Noise statistics generation**: ~10-20 minutes per mask (1M samples)
- **Total**: ~30-40 minutes on GPU

## How It Works

### 1. DCT Analysis of CIFAR-10

For each 32×32 CIFAR-10 image:
1. Convert RGB → YCrCb color space (separates luminance and chrominance)
2. Divide into 16 non-overlapping 8×8 patches
3. Apply 2D DCT to each patch
4. Accumulate statistics across all images and patches

**Result**: Mean and variance for each of the 64 DCT coefficients in each channel.

### 2. Mask Creation

For each DCT coefficient position (i, j):
- If `|mean[i,j]| < 0.1` AND `variance[i,j] < threshold`:
  - Set `mask[i,j] = True` (will be zeroed out)
- Otherwise:
  - Set `mask[i,j] = False` (keep this frequency)

**Intuition**: Low-mean, low-variance coefficients carry little information in natural images, so we can safely mask them.

### 3. DCT-Masked Sampling

To generate a sample from the DCT-masked source:
1. Sample Gaussian noise: `x ~ N(0, I)`
2. Convert to YCrCb color space
3. Apply DCT to 8×8 patches
4. Zero out masked coefficients: `DCT[mask] = 0`
5. Apply inverse DCT
6. Convert back to RGB

This creates structured noise that respects natural image frequency characteristics.

### 4. Normalization

The DCT masking operation changes the distribution of the noise. To use it effectively in flow matching:
1. Generate many samples through the DCT masking process
2. Compute empirical mean and std
3. Normalize: `x_normalized = (x - mean) / std`

This ensures the source distribution has zero mean and unit variance like standard Gaussian.

## Visualization

See `fourier.ipynb` for visualizations including:

**DCT Coefficient Heatmaps**:
```
Luminance Mean          Luminance Std
[128.0  0.1 -0.1 ...]   [12.3  5.4  3.2 ...]
...                     ...
```

The heatmaps show which frequency components have low variance (candidates for masking).

## Usage in Flow Matching

To use DCT-masked source in your flow matching model:

```python
from utils.dct import DCT_TORCH

# Initialize with weak masking
dct = DCT_TORCH(
    span=3,
    dct_stats=torch.load('data/noise_dctmask6_weak_torch.pt'),
    device='cuda',
    mask='weak',
    batch_size=128,
    root='data/'
)

# Sample from DCT-masked source
x0 = dct.sample(batch_size)  # Returns (B, 3, 32, 32) normalized tensor
```

## References

- JPEG compression uses similar DCT-based frequency masking with quantization tables
- DCT masking provides a structured source distribution that can improve flow matching training
- See paper pages 22-23 for theoretical motivation

## Notes

- **Weak vs Strong**: Weak masking is more conservative (masks fewer coefficients), strong masking is more aggressive. Choose based on your application.
- **RGB vs YCrCb**: The code works in YCrCb space because it separates luminance (Y) and chrominance (Cb, Cr), which have different frequency characteristics.
- **8×8 blocks**: Following JPEG convention, we use 8×8 blocks for DCT. The (0,0) coefficient is DC (average), and higher indices are higher frequencies.
