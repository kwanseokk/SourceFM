"""
Gaussian Mixture Model (GMM) Fitting Script

This script fits a Gaussian Mixture Model to CIFAR-10 training data and computes
normalization statistics. The GMM provides a more flexible source distribution
than standard Gaussian noise for flow matching models.

Outputs:
    - cifar10_gmm{n_components}_pca{n_pca}_{covariance_type}.pkl
    - cifar10_gmm{n_components}_pca{n_pca}_{covariance_type}_stats.pt
"""

import os
import sys
import argparse
import torch
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Add parent directory to path to import GMM_TORCH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.gmm import GMM_TORCH


def load_cifar10_data(data_root='./data', use_augmentation=False):
    """
    Load CIFAR-10 training dataset.

    Args:
        data_root: Root directory for CIFAR-10 data
        use_augmentation: Whether to use data augmentation (random flip)

    Returns:
        numpy array of shape (50000, 3072) containing flattened images
    """
    print(f"Loading CIFAR-10 from {data_root}...")

    transform_list = []
    if use_augmentation:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.Compose(transform_list),
    )

    # Load all images at once
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=len(trainset),
        shuffle=False
    )

    images, labels = next(iter(trainloader))

    # Flatten images: (N, 3, 32, 32) -> (N, 3072)
    images_np = images.numpy().reshape(images.shape[0], -1)

    print(f"Loaded {len(images_np)} images with shape {images_np.shape}")

    return images_np


def fit_gmm(data, n_components, covariance_type='full', random_state=0):
    """
    Fit a Gaussian Mixture Model to the data.

    Args:
        data: numpy array of shape (N, D)
        n_components: Number of Gaussian components
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        random_state: Random seed for reproducibility

    Returns:
        Fitted GMM model
    """
    print(f"\nFitting GMM with {n_components} components...")
    print(f"  Covariance type: {covariance_type}")
    print(f"  Data shape: {data.shape}")

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )

    gmm.fit(data)

    print(f"  Converged: {gmm.converged_}")
    print(f"  Number of iterations: {gmm.n_iter_}")
    print(f"  Log-likelihood: {gmm.score(data):.2f}")

    return gmm


def fit_pca(data, n_components):
    """
    Fit PCA to the data.

    Args:
        data: numpy array of shape (N, D)
        n_components: Number of principal components

    Returns:
        Fitted PCA model
    """
    if n_components >= data.shape[1]:
        print(f"\nSkipping PCA (n_components={n_components} >= data_dim={data.shape[1]})")
        return None

    print(f"\nFitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca.fit(data)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"  Explained variance: {explained_var:.4f}")

    return pca


def compute_gmm_statistics(gmm, n_samples=50000, batch_size=128, device='cuda'):
    """
    Compute normalization statistics for GMM samples.

    Args:
        gmm: Fitted sklearn GMM model
        n_samples: Number of samples to generate
        batch_size: Batch size for sampling
        device: Device to use for computation

    Returns:
        Dictionary with 'mean' and 'std' tensors
    """
    print(f"\nComputing GMM statistics from {n_samples} samples...")

    # Convert GMM to PyTorch format
    gmm_torch = GMM_TORCH(gmm=gmm, dtype=torch.float32, device=device)

    samples = []
    num_batches = (n_samples // batch_size) + 1

    for _ in tqdm(range(num_batches), desc="Sampling"):
        samples.append(gmm_torch.sample(batch_size))

    samples = torch.cat(samples, dim=0)[:n_samples]  # Trim to exact size

    # Compute statistics
    mean = samples.mean(0, keepdim=True)
    std = samples.std(0, keepdim=True)

    print(f"  Sample mean: {mean.mean().item():.6f}")
    print(f"  Sample std: {std.mean().item():.6f}")

    return {'mean': mean, 'std': std}


def main():
    parser = argparse.ArgumentParser(description='Fit GMM to CIFAR-10 data')
    parser.add_argument('--output_dir', type=str, default='../../data',
                        help='Directory to save output files')
    parser.add_argument('--data_root', type=str, default='../../data',
                        help='Root directory for CIFAR-10 data')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of GMM components')
    parser.add_argument('--n_pca', type=int, default=3072,
                        help='Number of PCA components (3072 = no PCA)')
    parser.add_argument('--covariance_type', type=str, default='full',
                        choices=['full', 'tied', 'diag', 'spherical'],
                        help='Type of covariance matrix')
    parser.add_argument('--n_stat_samples', type=int, default=50000,
                        help='Number of samples for computing statistics')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for sampling')
    parser.add_argument('--random_state', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--augmentation', action='store_true',
                        help='Use random horizontal flip augmentation')

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    images_np = load_cifar10_data(args.data_root, args.augmentation)

    # Apply PCA if requested
    pca = None
    if args.n_pca < images_np.shape[1]:
        pca = fit_pca(images_np, args.n_pca)
        images_pca = pca.transform(images_np)
        data_for_gmm = images_pca
    else:
        data_for_gmm = images_np

    # Fit GMM
    gmm = fit_gmm(
        data_for_gmm,
        args.n_components,
        args.covariance_type,
        args.random_state
    )

    # Save GMM and PCA models
    model_filename = f'cifar10_gmm{args.n_components}_pca{args.n_pca}_{args.covariance_type}.pkl'
    model_path = os.path.join(output_dir, model_filename)

    joblib.dump({'gmm': gmm, 'pca': pca}, model_path)
    print(f"\nSaved model to: {model_path}")

    # Compute statistics
    stats = compute_gmm_statistics(
        gmm,
        args.n_stat_samples,
        args.batch_size,
        device
    )

    # Save statistics
    stats_filename = f'cifar10_gmm{args.n_components}_pca{args.n_pca}_{args.covariance_type}_stats.pt'
    stats_path = os.path.join(output_dir, stats_filename)

    torch.save(stats, stats_path)
    print(f"Saved statistics to: {stats_path}")

    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == '__main__':
    main()
