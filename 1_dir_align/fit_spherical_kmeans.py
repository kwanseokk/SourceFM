"""
Spherical K-Means Clustering with Directional Alignment

This script performs spherical k-means clustering on CIFAR-10 data with optional
horizontal flip augmentation. It includes elbow study with silhouette scores to
find the optimal number of clusters.

Outputs:
    - spherical_kmeans{K}_flip.pt: Full clustering results with vMF parameters
    - clusters_spherical_kmeans{K}_flip.pt: Cluster assignments only
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_and_preprocess_cifar10(augment_flip=True, data_root='../data'):
    """
    Load CIFAR-10 and preprocess to unit hypersphere.

    Args:
        augment_flip: Whether to include horizontal flip augmentation
        data_root: Root directory for CIFAR-10 data

    Returns:
        data: Normalized vectors on unit sphere (N, 3072) or (2N, 3072) if flipped
        labels: Corresponding labels
    """
    # Transform: ToTensor + Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform)

    # Extract images and labels
    imgs = torch.stack([img for img, _ in full_dataset])  # (N, 3, 32, 32)
    labels = torch.tensor([lbl for _, lbl in full_dataset])  # (N,)

    N = imgs.size(0)

    if augment_flip:
        # Flatten and normalize original images
        flat_imgs = imgs.view(N, -1)  # (N, 3072)
        flat_imgs = F.normalize(flat_imgs, p=2, dim=1)

        # Create horizontally flipped version
        flipped = imgs.flip(dims=[3])  # Flip width dimension
        flipped = flipped.view(N, -1)  # (N, 3072)
        flipped = F.normalize(flipped, p=2, dim=1)

        # Concatenate original + flipped
        data = torch.cat([flat_imgs, flipped], dim=0)  # (2N, 3072)
        labels = torch.cat([labels, labels], dim=0)  # (2N,)
    else:
        # Just flatten and normalize
        data = imgs.view(N, -1)  # (N, 3072)
        data = F.normalize(data, p=2, dim=1)

    print(f"Loaded CIFAR-10 with shape: {data.shape}")
    return data, labels


def spherical_kmeans(X, n_clusters=10, max_iter=1000, tol=1e-4, device='cuda'):
    """
    Spherical K-Means clustering using cosine similarity.

    Uses k-means++ initialization with cosine distance.

    Args:
        X: Data tensor (N, D), already L2-normalized
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        device: Device to use

    Returns:
        centroids: Cluster centroids (K, D), L2-normalized
        assignments: Cluster assignments (N,)
    """
    X = X.to(device)
    n_samples, n_features = X.shape

    # K-means++ initialization
    centroids = []
    first_idx = torch.randint(0, n_samples, (1,)).item()
    centroids.append(X[first_idx])

    for _ in range(n_clusters - 1):
        # Compute distances to nearest centroid
        sims = torch.mm(X, torch.stack(centroids).T)  # (N, k)
        max_sim, _ = sims.max(dim=1)  # (N,)
        dists = 1 - max_sim  # Cosine distance
        probs = dists ** 2  # k-means++: squared distance
        probs = torch.clamp(probs, min=1e-12)
        probs /= probs.sum()

        next_idx = torch.multinomial(probs, 1).item()
        centroids.append(X[next_idx])

    centroids = torch.stack(centroids)  # (K, D)

    # K-Means iterations
    for iteration in range(max_iter):
        # Assignment step: assign to nearest centroid by cosine similarity
        sims = torch.mm(X, centroids.T)  # (N, K)
        assignments = sims.argmax(dim=1)  # (N,)

        # Update step: compute mean and normalize
        new_centroids = []
        for k in range(n_clusters):
            mask = (assignments == k)
            if mask.sum() == 0:
                # Keep previous centroid if cluster is empty
                new_centroids.append(centroids[k])
            else:
                pts = X[mask]  # (n_k, D)
                c = pts.mean(dim=0, keepdim=True)  # (1, D)
                c = F.normalize(c, p=2, dim=1)[0]  # L2 normalize
                new_centroids.append(c)

        new_centroids = torch.stack(new_centroids)  # (K, D)

        # Check convergence
        shift = 1 - torch.cosine_similarity(centroids, new_centroids, dim=1).mean()
        centroids = new_centroids

        if shift < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return centroids.cpu(), assignments.cpu()


def estimate_kappa(r_bar, dim, method='sra'):
    """
    Estimate concentration parameter kappa for von Mises-Fisher distribution.

    Args:
        r_bar: Mean resultant length (mean cosine similarity to centroid)
        dim: Dimensionality
        method: Estimation method ('banerjee', 'sra', 'newton')

    Returns:
        kappa: Concentration parameter
    """
    r_bar = torch.clamp(r_bar, min=1e-6, max=0.999)

    if method == 'banerjee':
        return (r_bar * (dim - r_bar**2)) / (1 - r_bar**2)
    elif method == 'sra':
        return (r_bar * (dim - 1) - r_bar**3) / (1 - r_bar**2 + 1e-10)
    elif method == 'newton':
        kappa = (r_bar * dim) / (1 - r_bar**2)  # Initial value
        for _ in range(500):
            a = (dim/2 - 1) * r_bar
            f = a / kappa + r_bar - (1 / torch.tanh(kappa) + 1/kappa)
            df = -a / (kappa**2) + (1 / torch.sinh(kappa)**2 - 1/(kappa**2))
            kappa -= f / df
        return kappa
    else:
        raise ValueError(f"Unknown method: {method}")


def fit_vmf(cluster_points, centroids, method='newton'):
    """
    Fit von Mises-Fisher distribution to cluster points.

    Args:
        cluster_points: Points in cluster (n, D)
        centroids: Cluster centroid (K, D)
        method: Kappa estimation method

    Returns:
        mu: Direction parameter (same as centroid)
        kappa: Concentration parameter
    """
    device = cluster_points.device
    cluster_points = F.normalize(cluster_points, p=2, dim=1)
    centroids = centroids.to(device)
    n_clusters, n_features = centroids.shape

    # μ is the centroid (already normalized)
    mu = centroids

    # Estimate κ from mean resultant length
    r_bar = torch.mm(cluster_points, mu.T).mean(dim=0)
    kappa = estimate_kappa(r_bar, n_features, method)

    return mu, kappa


def elbow_study(data, k_range, device='cuda', n_init=10):
    """
    Perform elbow study to find optimal number of clusters.

    Args:
        data: Normalized data (N, D)
        k_range: Range of K values to try
        device: Device to use
        n_init: Number of random initializations per K

    Returns:
        results: Dictionary with K values, silhouette scores, and inertias
    """
    print("\nPerforming elbow study with silhouette scores...")

    k_values = []
    silhouette_scores = []
    inertias = []

    data_np = data.cpu().numpy()

    for k in tqdm(k_range, desc="Elbow study"):
        best_score = -1
        best_inertia = float('inf')
        best_assignments = None

        # Run multiple initializations
        for _ in range(n_init):
            centroids, assignments = spherical_kmeans(
                data, n_clusters=k, max_iter=500, device=device)

            # Compute silhouette score
            score = silhouette_score(data_np, assignments.numpy(), metric='cosine')

            # Compute inertia (sum of cosine distances)
            sims = torch.mm(data, centroids.T)
            assigned_sims = sims[torch.arange(len(assignments)), assignments]
            inertia = (1 - assigned_sims).sum().item()

            if score > best_score:
                best_score = score
                best_inertia = inertia
                best_assignments = assignments

        k_values.append(k)
        silhouette_scores.append(best_score)
        inertias.append(best_inertia)

        print(f"  K={k}: Silhouette={best_score:.4f}, Inertia={best_inertia:.2f}")

    return {
        'k_values': k_values,
        'silhouette_scores': silhouette_scores,
        'inertias': inertias
    }


def plot_elbow_study(results, output_path):
    """Plot elbow study results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Silhouette scores
    ax1.plot(results['k_values'], results['silhouette_scores'], 'o-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs K')
    ax1.grid(True, alpha=0.3)

    # Inertias
    ax2.plot(results['k_values'], results['inertias'], 'o-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Inertia (Sum of Cosine Distances)')
    ax2.set_title('Inertia vs K')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved elbow study plot to: {output_path}")


def compute_angle_statistics(cluster_points, centroids):
    """
    Compute angle statistics for each cluster.

    Args:
        cluster_points: List of tensors, one per cluster
        centroids: Cluster centroids (K, D)

    Returns:
        theta_min: Minimum angle per cluster (K,)
        theta_max: Maximum angle per cluster (K,)
        compressed_angles: Quantile-compressed angles (K, 100)
    """
    n_clusters = len(cluster_points)
    K_quantiles = 100

    theta_min = []
    theta_max = []
    compressed_angles = []

    for k in range(n_clusters):
        # Compute angles between cluster points and centroid
        sims = torch.mm(cluster_points[k], centroids[k].unsqueeze(0).T).squeeze(1)
        sims = torch.clamp(sims, -1 + 1e-6, 1 - 1e-6)
        angles = torch.acos(sims)

        theta_min.append(angles.min())
        theta_max.append(angles.max())

        # Compress angles using quantiles
        sorted_angles = np.sort(angles.cpu().numpy())
        quantiles = np.linspace(0, 1, K_quantiles)
        angles_compressed = np.quantile(sorted_angles, quantiles)
        compressed_angles.append(torch.tensor(angles_compressed))

    theta_min = torch.stack(theta_min)
    theta_max = torch.stack(theta_max)
    compressed_angles = torch.stack(compressed_angles)

    return theta_min, theta_max, compressed_angles


def main():
    parser = argparse.ArgumentParser(description='Spherical K-Means Clustering for CIFAR-10')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='Number of clusters (default: 3 for spherical_kmeans3_flip.pt)')
    parser.add_argument('--augment_flip', action='store_true', default=True,
                        help='Include horizontal flip augmentation')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory for CIFAR-10 data')
    parser.add_argument('--output_dir', type=str, default='../data',
                        help='Directory to save output files')
    parser.add_argument('--elbow_study', action='store_true',
                        help='Perform elbow study before clustering')
    parser.add_argument('--elbow_min', type=int, default=2,
                        help='Minimum K for elbow study')
    parser.add_argument('--elbow_max', type=int, default=10,
                        help='Maximum K for elbow study')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Maximum iterations for k-means')
    parser.add_argument('--kappa_method', type=str, default='newton',
                        choices=['banerjee', 'sra', 'newton'],
                        help='Method for estimating kappa')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). If None, auto-detect.')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    print("\nLoading CIFAR-10...")
    data, labels = load_and_preprocess_cifar10(args.augment_flip, args.data_root)

    # Elbow study (optional)
    if args.elbow_study:
        k_range = range(args.elbow_min, args.elbow_max + 1)
        results = elbow_study(data, k_range, device)

        # Plot and save results
        plot_path = os.path.join(args.output_dir, 'elbow_study.png')
        plot_elbow_study(results, plot_path)

        # Find best K by silhouette score
        best_idx = np.argmax(results['silhouette_scores'])
        best_k = results['k_values'][best_idx]
        print(f"\nBest K by silhouette score: {best_k}")
        print("You can re-run with --n_clusters {best_k} to use this value")

    # Perform clustering with specified K
    print(f"\nPerforming spherical k-means with K={args.n_clusters}...")
    centroids, assignments = spherical_kmeans(
        data, n_clusters=args.n_clusters, max_iter=args.max_iter, device=device)

    print(f"Cluster centroids shape: {centroids.shape}")

    # Compute silhouette score
    data_np = data.cpu().numpy()
    labels_np = assignments.cpu().numpy()
    sil_score = silhouette_score(data_np, labels_np, metric='cosine')
    print(f"Silhouette Score: {sil_score:.4f}")

    # Group points by cluster
    cluster_points = []
    for k in range(args.n_clusters):
        cluster_points.append(data[assignments == k])
        print(f"Cluster {k}: {cluster_points[k].shape[0]} points")

    # Fit vMF distributions
    print(f"\nFitting vMF distributions using method: {args.kappa_method}...")
    mu_list = []
    kappa_list = []
    for k in range(args.n_clusters):
        mu, kappa = fit_vmf(cluster_points[k], centroids[k].unsqueeze(0), method=args.kappa_method)
        mu_list.append(mu)
        kappa_list.append(kappa)

    centroid_tensors = torch.cat(mu_list, dim=0)
    kappa_tensors = torch.cat(kappa_list, dim=0)

    # Compute sample ratios (cluster proportions)
    sample_ratio = torch.tensor([cp.shape[0] for cp in cluster_points], dtype=torch.float32)
    sample_ratio = sample_ratio / sample_ratio.sum()

    # Compute per-cluster normalization statistics
    print("\nComputing per-cluster normalization statistics...")
    mus = []
    stds = []
    for i in range(args.n_clusters):
        samples = cluster_points[i].reshape(-1, 3, 32, 32)
        norm_mu, norm_std = samples.mean(dim=[0, 2, 3], keepdims=True), samples.std(dim=[0, 2, 3], keepdims=True)
        mus.append(norm_mu)
        stds.append(norm_std)

    norm_mu = torch.stack(mus, dim=0)  # Shape: (K, 1, 3, 1, 1)
    norm_std = torch.stack(stds, dim=0)  # Shape: (K, 1, 3, 1, 1)
    all_norm_std = data.reshape(-1, 3, 32, 32).std()  # Scalar

    # Compute angle statistics
    print("\nComputing angle statistics...")
    theta_min, theta_max, compressed_angles = compute_angle_statistics(cluster_points, centroid_tensors)

    # Save results
    flip_suffix = '_flip' if args.augment_flip else ''
    output_filename = f'spherical_kmeans{args.n_clusters}{flip_suffix}.pt'
    output_path = os.path.join(args.output_dir, output_filename)

    torch.save({
        'centroid': centroid_tensors,
        'assignments': assignments,
        'kappa': kappa_tensors,
        'emp_angles': compressed_angles,
        'theta_min': theta_min,
        'theta_max': theta_max,
        'sample_ratio': sample_ratio,
        'norm_mu': norm_mu,
        'norm_std': norm_std,
        'all_norm_std': all_norm_std,
    }, output_path)

    print(f"\nSaved clustering results to: {output_path}")

    # Also save just assignments
    assignments_filename = f'clusters_spherical_kmeans{args.n_clusters}{flip_suffix}.pt'
    assignments_path = os.path.join(args.output_dir, assignments_filename)
    torch.save(assignments, assignments_path)
    print(f"Saved cluster assignments to: {assignments_path}")

    # Print summary
    print("\n" + "="*60)
    print("Clustering Summary")
    print("="*60)
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Silhouette score: {sil_score:.4f}")
    print(f"Sample ratios: {sample_ratio.tolist()}")
    print(f"Kappa values: {kappa_tensors.squeeze().tolist()}")
    print("="*60)


if __name__ == '__main__':
    main()
