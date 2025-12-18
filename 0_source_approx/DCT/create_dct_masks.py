"""
DCT Mask Creation Script

This script creates DCT frequency masks for weak and strong compression levels.
It analyzes CIFAR-10 training images to determine which DCT frequency components
can be masked (zeroed out) without significant information loss.

Outputs:
    - luminance_mask_weak.npy, luminance_mask_strong.npy
    - chrominance_cb_mask_weak.npy, chrominance_cb_mask_strong.npy
    - chrominance_cr_mask_weak.npy, chrominance_cr_mask_strong.npy
    - luminance_mean.npy, luminance_var.npy
    - chrominance_cb_mean.npy, chrominance_cb_var.npy
    - chrominance_cr_mean.npy, chrominance_cr_var.npy
    - noise_dctmask6_weak_torch.pt, noise_dctmask6_strong_torch.pt
"""

import numpy as np
import cv2
import torch
import sys
import os
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from scipy.fftpack import dct as scipy_dct

# Add parent directory to path to import DCT_TORCH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.dct import DCT_TORCH


def rgb_to_ycrcb(rgb):
    """
    Convert RGB image to YCrCb color space.
    Args:
        rgb: numpy array of shape (H, W, 3) with values in [0, 1]
    Returns:
        ycrcb: numpy array of shape (H, W, 3)
    """
    # Scale to 0-255 and convert to uint8
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    # Use OpenCV for color conversion
    ycrcb = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb)
    return ycrcb


def compute_dct_statistics(images):
    """
    Compute DCT statistics (mean and variance) for each frequency component.

    Args:
        images: numpy array of shape (N, 3, 32, 32) with values in [0, 1]

    Returns:
        Dictionary containing mean and variance for each channel
    """
    num_images = len(images)
    num_patches = (32 // 8) ** 2  # 16 patches per image

    # Initialize accumulators for each channel
    lum_coeffs = []
    cb_coeffs = []
    cr_coeffs = []

    print("Computing DCT coefficients...")
    for img in tqdm(images):
        # Convert from (3, 32, 32) to (32, 32, 3)
        img_hwc = img.transpose(1, 2, 0)

        # Convert to YCrCb
        ycrcb = rgb_to_ycrcb(img_hwc)

        # Process each 8x8 patch
        for i in range(0, 32, 8):
            for j in range(0, 32, 8):
                patch_y = ycrcb[i:i+8, j:j+8, 0].astype(np.float32)
                patch_cb = ycrcb[i:i+8, j:j+8, 1].astype(np.float32)
                patch_cr = ycrcb[i:i+8, j:j+8, 2].astype(np.float32)

                # Apply DCT using scipy
                dct_y = scipy_dct(scipy_dct(patch_y.T, norm='ortho').T, norm='ortho')
                dct_cb = scipy_dct(scipy_dct(patch_cb.T, norm='ortho').T, norm='ortho')
                dct_cr = scipy_dct(scipy_dct(patch_cr.T, norm='ortho').T, norm='ortho')

                lum_coeffs.append(dct_y)
                cb_coeffs.append(dct_cb)
                cr_coeffs.append(dct_cr)

    # Convert to numpy arrays
    lum_coeffs = np.array(lum_coeffs)  # Shape: (N * 16, 8, 8)
    cb_coeffs = np.array(cb_coeffs)
    cr_coeffs = np.array(cr_coeffs)

    # Compute mean and variance across all patches
    stats = {
        'luminance': {
            'mean': np.mean(lum_coeffs, axis=0),
            'var': np.var(lum_coeffs, axis=0)
        },
        'chrominance_cb': {
            'mean': np.mean(cb_coeffs, axis=0),
            'var': np.var(cb_coeffs, axis=0)
        },
        'chrominance_cr': {
            'mean': np.mean(cr_coeffs, axis=0),
            'var': np.var(cr_coeffs, axis=0)
        }
    }

    return stats


def create_masks(stats, mean_threshold=0.1, var_threshold_weak=2.0, var_threshold_strong=4.0):
    """
    Create masks based on mean and variance thresholds.

    Args:
        stats: Dictionary containing mean and variance for each channel
        mean_threshold: Threshold for absolute mean
        var_threshold_weak: Variance threshold for weak mask
        var_threshold_strong: Variance threshold for strong mask

    Returns:
        Dictionary containing weak and strong masks for each channel
    """
    masks = {}

    for channel in ['luminance', 'chrominance_cb', 'chrominance_cr']:
        mean = stats[channel]['mean']
        var = stats[channel]['var']

        # Create masks: True where we want to mask (zero out)
        # Mask where absolute mean is small AND variance is small
        mask_weak = (np.abs(mean) < mean_threshold) & (var < var_threshold_weak)
        mask_strong = (np.abs(mean) < mean_threshold) & (var < var_threshold_strong)

        masks[f'{channel}_weak'] = mask_weak
        masks[f'{channel}_strong'] = mask_strong

    return masks


def generate_noise_statistics(mask_strength, output_dir, num_samples=1000000, batch_size=500):
    """
    Generate noise statistics using the DCT_TORCH class.
    This follows the approach in dctmask.py.

    Args:
        mask_strength: 'weak' or 'strong'
        output_dir: Directory where masks are saved
        num_samples: Number of noise samples to generate
        batch_size: Batch size for processing

    Returns:
        Dictionary with 'mean' and 'std' tensors
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    span = 3

    # Initialize DCT_TORCH
    print(f"\nInitializing DCT_TORCH for {mask_strength} mask...")
    dct = DCT_TORCH(
        span=span,
        dct_stats=None,
        device=device,
        mask=mask_strength,
        batch_size=batch_size,
        rgb=False,
        root=output_dir
    )

    # Sample noise
    samples = []
    print(f"Generating {num_samples} noise samples...")
    for i in tqdm(range(num_samples // batch_size)):
        x = dct.sample(batch_size)
        samples.append(x)

    samples = torch.cat(samples, dim=0)

    # Compute statistics
    sample_stat = {
        'mean': samples.mean(0),
        'std': samples.std(0)
    }

    return sample_stat


def main():
    parser = argparse.ArgumentParser(description='Create DCT masks from CIFAR-10')
    parser.add_argument('--output_dir', type=str, default='../../data',
                        help='Directory to save output files')
    parser.add_argument('--mean_threshold', type=float, default=0.1,
                        help='Threshold for absolute mean')
    parser.add_argument('--var_threshold_weak', type=float, default=2.0,
                        help='Variance threshold for weak mask')
    parser.add_argument('--var_threshold_strong', type=float, default=4.0,
                        help='Variance threshold for strong mask')
    parser.add_argument('--num_noise_samples', type=int, default=1000000,
                        help='Number of noise samples for computing statistics')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for processing')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    trainset = datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    images, _ = next(iter(trainloader))
    images_np = images.numpy()

    print(f"Loaded {len(images_np)} images")

    # Compute DCT statistics
    stats = compute_dct_statistics(images_np)

    # Save mean and variance
    print("\nSaving mean and variance statistics...")
    for channel in ['luminance', 'chrominance_cb', 'chrominance_cr']:
        mean_path = os.path.join('stats', f'{channel}_mean.npy')
        var_path = os.path.join('stats', f'{channel}_var.npy')
        np.save(mean_path, stats[channel]['mean'])
        np.save(var_path, stats[channel]['var'])
        print(f"  Saved {mean_path}")
        print(f"  Saved {var_path}")

    # Create masks
    print("\nCreating masks...")
    masks = create_masks(stats, args.mean_threshold, args.var_threshold_weak, args.var_threshold_strong)

    # Save masks
    print("\nSaving masks...")
    for mask_name, mask_data in masks.items():
        mask_path = os.path.join(output_dir, f'{mask_name}.npy')
        np.save(mask_path, mask_data)
        print(f"  {mask_name}: {mask_data.sum()}/{mask_data.size} coefficients masked")
        print(f"  Saved {mask_path}")

    # Generate noise statistics using DCT_TORCH
    print("\n" + "="*50)
    print("Generating noise statistics...")
    print("="*50)

    for mask_strength in ['weak', 'strong']:
        noise_stats = generate_noise_statistics(
            mask_strength,
            output_dir,
            args.num_noise_samples,
            args.batch_size
        )

        output_path = os.path.join(output_dir, f'noise_dctmask6_{mask_strength}_torch.pt')
        torch.save(noise_stats, output_path)
        print(f"\nSaved {output_path}")
        print(f"  Mean shape: {noise_stats['mean'].shape}")
        print(f"  Std shape: {noise_stats['std'].shape}")

    print("\n" + "="*50)
    print("Done! All files created successfully.")
    print("="*50)


if __name__ == '__main__':
    main()
