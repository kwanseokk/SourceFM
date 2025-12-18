import torch
import numpy as np
from typing import Optional, Union, List

class SphericalKNN_Sampler:
    def __init__(self, sknn, method='vmf', normalization=True, dtype=torch.float32, device="cpu"):
        """
        Convert a spherical k-nearest neighbors (sknn) model into a PyTorch-compatible format.

        Args:
            sknn (dict): Dictionary containing model parameters (centroid, kappa, etc.).
            method (str): Sampling method ('vmf', 'uni', or 'emp').
            dtype (torch.dtype): Data type for PyTorch tensors.
            device (str or torch.device): Device to store tensors (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.dtype = dtype
        self.method = method
        self.centroid = sknn['centroid'].to(device=device, dtype=dtype)
        self.kappa = sknn['kappa'].to(device=device, dtype=dtype)
        self.emp_angles = sknn['emp_angles'].to(device=device, dtype=dtype)
        self.sample_ratio = sknn['sample_ratio'].to(device=device, dtype=dtype)
        self.norm_mu = sknn['norm_mu'].to(device=device, dtype=dtype)
        self.norm_std = sknn['norm_std'].to(device=device, dtype=dtype)
        self.theta_max = sknn['theta_max'].to(device=device, dtype=dtype)
        self.n_cluster = self.centroid.shape[0]
        self.n_emp_angles = self.emp_angles.shape[0]
        
        self.normalization = normalization
        
    def sample(self, 
               num_samples: Optional[int] = None, 
               cluster_number: Optional[Union[int, List[int], torch.Tensor]] = None,
               return_cluster_numbers: bool = False
               ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample points from the spherical distribution.

        Args:
            num_samples (Optional[int]): Number of samples to generate.
            cluster_number (Optional[Union[int, List[int], torch.Tensor]]): Cluster indices to sample from.
                - If None, cluster indices are sampled from self.sample_ratio.
                - If an int, all samples come from that cluster.
                - If a list or torch.Tensor, the number of samples will be deduced from its length.
            normalization (bool): If True, apply per-channel normalization using cluster-specific norm_mu and norm_std.
            return_cluster_numbers (bool): If True, return the cluster numbers along with the samples.
        
        Returns:
            torch.Tensor or tuple: Samples of shape (num_samples, 3, 32, 32), and optionally the cluster numbers.
        """
        if num_samples is None and cluster_number is None:
            raise ValueError("Either num_samples or cluster_number must be provided.")
        
        if num_samples is not None:
            if cluster_number is None:
                cluster_numbers = torch.multinomial(self.sample_ratio, num_samples=num_samples, replacement=True)
            elif isinstance(cluster_number, int):
                cluster_numbers = torch.full((num_samples,), cluster_number, dtype=torch.int, device=self.device)
            elif isinstance(cluster_number, list):
                cluster_numbers = torch.tensor(cluster_number, dtype=torch.int, device=self.device)
                if cluster_numbers.numel() != num_samples:
                    raise ValueError("The length of the cluster_number list must equal num_samples.")
            elif isinstance(cluster_number, torch.Tensor):
                if cluster_number.dtype != torch.int:
                    cluster_numbers = cluster_number.to(dtype=torch.int, device=self.device)
                else:
                    cluster_numbers = cluster_number.to(device=self.device)
                if cluster_numbers.numel() != num_samples:
                    raise ValueError("The cluster_number tensor must have num_samples elements.")
            else:
                raise ValueError("Invalid type for cluster_number; must be None, int, list, or torch.Tensor.")
        else:
            if isinstance(cluster_number, int):
                num_samples = 1
                cluster_numbers = torch.tensor([cluster_number], dtype=torch.int, device=self.device)
            elif isinstance(cluster_number, list):
                num_samples = len(cluster_number)
                cluster_numbers = torch.tensor(cluster_number, dtype=torch.int, device=self.device)
            elif isinstance(cluster_number, torch.Tensor):
                num_samples = cluster_number.numel()
                if cluster_number.dtype != torch.int:
                    cluster_numbers = cluster_number.to(dtype=torch.int, device=self.device)
                else:
                    cluster_numbers = cluster_number.to(device=self.device)
            else:
                raise ValueError("Invalid type for cluster_number; must be int, list, or torch.Tensor.")

        if self.method == 'vmf':
            x = self._sample_vmf(cluster_numbers)
        elif self.method == 'uni':
            x = self._sample_cap_uniform_angle(cluster_numbers)
        elif self.method == 'emp':
            x = self._sample_empirical_angle(cluster_numbers)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

        x = x.reshape(-1, 3, 32, 32)
        if self.normalization:
            # Select cluster-specific norm_mu and norm_std
            norm_mu = self.norm_mu[cluster_numbers].squeeze(1)  # Shape: (num_samples, 3, 1, 1)
            norm_std = self.norm_std[cluster_numbers].squeeze(1)  # Shape: (num_samples, 3, 1, 1)
            x = x / norm_std
        else:
            x = x * 59.4109

        if return_cluster_numbers:
            return x, cluster_numbers
        return x
            
    def rotation_matrix_to(self, target_dir):
        """
        Compute a Householder reflection matrix to map e_d to target_dir.

        Args:
            target_dir (torch.Tensor): Target direction vector.

        Returns:
            torch.Tensor: Reflection matrix.
        """
        dim = target_dir.shape[0]
        device = target_dir.device
        
        base_dir = torch.zeros(dim, device=device)
        base_dir[-1] = 1.0
        
        if torch.allclose(target_dir, base_dir, atol=1e-6):
            return torch.eye(dim, device=device)
        
        if torch.allclose(target_dir, -base_dir, atol=1e-6):
            return -torch.eye(dim, device=device)
        
        v = target_dir - base_dir
        v = v / torch.norm(v)
        R = torch.eye(dim, device=device) - 2 * torch.outer(v, v)
        return R

    def _compute_angular_distances(self, samples: torch.Tensor, cluster_numbers: torch.Tensor) -> torch.Tensor:
        """
        Compute angular distances between samples and their corresponding cluster centroids.

        Args:
            samples (torch.Tensor): Samples of shape (num_samples, d) or (num_samples, 3, 32, 32).
            cluster_numbers (torch.Tensor): Cluster indices of shape (num_samples,).

        Returns:
            torch.Tensor: Angular distances in radians of shape (num_samples,).
        """
        if samples.dim() == 4:
            samples = samples.view(samples.size(0), -1)
        if samples.dim() != 2:
            raise ValueError("samples must be 2D or 4D tensor")
        num_samples, d = samples.shape
        if cluster_numbers.numel() != num_samples:
            raise ValueError("cluster_numbers must have the same number of elements as samples")
        if self.centroid.shape[1] != d:
            raise ValueError(f"samples must have dimension {self.centroid.shape[1]}, got {d}")
        
        # Check if samples are on the unit sphere
        samples_norm = torch.norm(samples, dim=1)
        if not torch.allclose(samples_norm, torch.ones(num_samples, device=samples.device), atol=1e-5):
            print("Warning: Some samples are not on the unit sphere")
        
        mu_batch = self.centroid[cluster_numbers]
        # Check if centroids are on the unit sphere
        mu_norm = torch.norm(mu_batch, dim=1)
        if not torch.allclose(mu_norm, torch.ones(num_samples, device=mu_batch.device), atol=1e-5):
            print("Warning: Some centroids are not on the unit sphere")
        
        cos_theta = (samples * mu_batch).sum(dim=1)
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        return theta

    def sample_and_check(self, num_samples):
        cluster_numbers = torch.multinomial(self.sample_ratio, num_samples=num_samples, replacement=True)
        samples = self.sample(num_samples=None, cluster_number=cluster_numbers)
        return self._compute_angular_distances(samples, cluster_numbers)
    
    def _sample_vmf(self, cluster_numbers: torch.Tensor) -> torch.Tensor:
        """
        Batched vMF sampling using Wood's algorithm.

        Args:
            cluster_numbers (torch.Tensor): Tensor of shape (B,) with cluster indices.

        Returns:
            torch.Tensor: Samples of shape (B, d).
        """
        mu_batch = self.centroid[cluster_numbers]
        kappa_batch = self.kappa[cluster_numbers]
        B, d = mu_batch.shape

        b = (-2 * kappa_batch + torch.sqrt(4 * kappa_batch**2 + (d - 1)**2)) / (d - 1)
        x0 = (1 - b) / (1 + b)
        c = kappa_batch * x0 + (d - 1) * torch.log(1 - x0**2)

        accepted = torch.zeros(B, dtype=torch.bool, device=self.device)
        w_result = torch.empty(B, device=self.device, dtype=self.dtype)

        max_iter = 10000
        iter_count = 0
        while not torch.all(accepted) and iter_count < max_iter:
            iter_count += 1
            pending_idx = (~accepted).nonzero(as_tuple=True)[0]
            N_pending = pending_idx.shape[0]
            b_pending = b[pending_idx].unsqueeze(1)
            x0_pending = x0[pending_idx].unsqueeze(1)
            c_pending = c[pending_idx].unsqueeze(1)
            kappa_pending = kappa_batch[pending_idx].unsqueeze(1)

            z = torch.rand(N_pending, 1, device=self.device, dtype=self.dtype)
            u = torch.rand(N_pending, 1, device=self.device, dtype=self.dtype)
            w_candidate = (1 - (1 + b_pending) * z) / (1 - (1 - b_pending) * z)
            crit = kappa_pending * w_candidate + (d - 1) * torch.log(1 - x0_pending * w_candidate) - c_pending
            accept_mask = crit >= torch.log(u)
            if accept_mask.any():
                accepted_indices = pending_idx[accept_mask.squeeze()]
                w_result[accepted_indices] = w_candidate[accept_mask].squeeze()
                accepted[accepted_indices] = True

        if iter_count >= max_iter and not torch.all(accepted):
            print("Warning: Rejection sampling reached max iterations, some samples may not be valid.")

        v = torch.randn(B, d - 1, device=self.device, dtype=self.dtype)
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        w_result = w_result.unsqueeze(1)
        samples = torch.cat([torch.sqrt(1 - w_result**2) * v, w_result], dim=1)

        e_d = torch.zeros(d, device=self.device, dtype=self.dtype)
        e_d[-1] = 1.0
        e_d_batch = e_d.unsqueeze(0).expand(B, d)
        u_reflect = e_d_batch - mu_batch
        u_norm = torch.norm(u_reflect, dim=1, keepdim=True) + 1e-12
        u_normalized = u_reflect / u_norm
        dot_prod = (samples * u_normalized).sum(dim=1, keepdim=True)
        samples_rot = samples - 2 * dot_prod * u_normalized
        samples_rot = torch.nn.functional.normalize(samples_rot, p=2, dim=1)
        return samples_rot

    def _sample_cap_uniform_angle(self, cluster_numbers: torch.Tensor) -> torch.Tensor:
        """
        Batched spherical cap sampling with uniform geodesic distance in [0, theta_max], where theta_max is cluster-specific.

        Args:
            cluster_numbers (torch.Tensor): Tensor of shape (B,) with cluster indices.

        Returns:
            torch.Tensor: Samples of shape (B, d).
        """
        mu_batch = self.centroid[cluster_numbers]
        theta_max = self.theta_max[cluster_numbers]
        B, d = mu_batch.shape

        U = torch.rand(B, device=self.device, dtype=self.dtype)
        r = theta_max * U

        v = torch.randn(B, d - 1, device=self.device, dtype=self.dtype)
        v = torch.nn.functional.normalize(v, p=2, dim=1)

        sin_r = torch.sin(r).unsqueeze(1)
        cos_r = torch.cos(r).unsqueeze(1)
        samples = torch.cat([v * sin_r, cos_r], dim=1)

        e_d = torch.zeros(d, device=self.device, dtype=self.dtype)
        e_d[-1] = 1.0
        e_d_batch = e_d.unsqueeze(0).expand(B, d)
        u_reflect = e_d_batch - mu_batch
        u_norm = torch.norm(u_reflect, dim=1, keepdim=True) + 1e-12
        u_normalized = u_reflect / u_norm
        dot_prod = (samples * u_normalized).sum(dim=1, keepdim=True)
        samples_rot = samples - 2 * dot_prod * u_normalized
        samples_rot = torch.nn.functional.normalize(samples_rot, p=2, dim=1)
        return samples_rot

    def _sample_empirical_angle(self, cluster_numbers: torch.Tensor) -> torch.Tensor:
        """
        Batched sampling mimicking empirical angle distributions per cluster.

        Args:
            cluster_numbers (torch.Tensor): Tensor of shape (B,) with cluster indices.

        Returns:
            torch.Tensor: Samples of shape (B, d).
        """
        mu_batch = self.centroid[cluster_numbers]
        B, d = mu_batch.shape
        results = torch.empty((B, d), device=self.device, dtype=self.dtype)

        unique_clusters, _ = torch.unique(cluster_numbers, sorted=True, return_inverse=True)
        for uc in unique_clusters:
            indices = (cluster_numbers == uc).nonzero(as_tuple=True)[0]
            n = indices.shape[0]
            mu_i = self.centroid[int(uc.item())]
            emp_angles = self.emp_angles[int(uc.item())]
            sorted_angles, sort_idx = torch.sort(emp_angles)
            N = sorted_angles.shape[0]
            quantiles = torch.linspace(0, 1, N, device=self.device, dtype=self.dtype)
            u = torch.rand(n, device=self.device, dtype=self.dtype)
            # Use torch.searchsorted for interpolation
            idx = torch.searchsorted(quantiles, u)
            idx = torch.clamp(idx, 0, N-1)
            sampled_angles = sorted_angles[idx]

            v = torch.randn(n, d - 1, device=self.device, dtype=self.dtype)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            
            sin_theta = torch.sin(sampled_angles).unsqueeze(1)
            cos_theta = torch.cos(sampled_angles).unsqueeze(1)
            samples_group = torch.cat([v * sin_theta, cos_theta], dim=1)
            R = self.rotation_matrix_to(mu_i)
            rotated_group = torch.nn.functional.normalize(samples_group @ R.T, p=2, dim=1)
            results[indices] = rotated_group
        return results
    