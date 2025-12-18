import torch
import numpy as np

def sample_vmf(mu: torch.Tensor, kappa: float, num_samples: int = 1) -> torch.Tensor:
    """
    Samples from the von Mises-Fisher distribution.

    Args:
        mu (torch.Tensor): The mean direction (must be a unit vector).
                           Shape: (dimension,)
        kappa (float): The concentration parameter (kappa >= 0).
        num_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: Samples from the von Mises-Fisher distribution.
                      Shape: (num_samples, dimension)
    """
    dimension = mu.size(0)
    device = mu.device
    dtype = mu.dtype

    # Ensure mu is a unit vector
    mu = mu / torch.linalg.norm(mu)

    if kappa < 1e-6:
        # For very small kappa, the distribution is approximately uniform on the sphere
        return _sample_uniform_sphere(dimension, num_samples, device=device, dtype=dtype)

    samples = torch.empty((num_samples, dimension), dtype=dtype, device=device)
    for i in range(num_samples):
        w = _sample_w(kappa, dimension, device=device, dtype=dtype)
        v = _sample_orthonormal_complement(mu.unsqueeze(0))
        v = v.squeeze(0)
        sample = w * mu + torch.sqrt(1.0 - w**2) * v
        samples[i] = sample

    return samples

def _sample_w(kappa: float, dimension: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Samples the cosine of the angle from the mean direction."""
    if dimension == 1:
        return torch.sign(torch.randn(1, device=device, dtype=dtype))
    elif dimension == 2:
        return (torch.rand(1, device=device, dtype=dtype) * 2 - 1)
    else:
        xi = kappa
        nu = torch.tensor((dimension - 1) / 2.0, device=device, dtype=dtype)
        rho = xi / (torch.sqrt(xi**2 + nu**2))
        s = (1 - rho**2)**0.5

        while True:
            u = torch.rand(1, device=device, dtype=dtype)
            z = torch.randn(1, device=device, dtype=dtype)
            v = s * z
            w = (rho + v) / torch.sqrt(1 + 2 * rho * v + v**2)
            if dimension == 3:
                accept = (kappa * w + (dimension - 1) * torch.log(1 - rho * w - v)) > \
                         (kappa * rho + (dimension - 1) * torch.log(1 - rho**2)) + torch.log(u)
            else:
                m = (dimension - 1) / 2.0
                accept = (kappa * w + (dimension - 1) * torch.log(1 - rho * w - v)) > \
                         (kappa * rho + (dimension - 1) * torch.log(1 - rho**2)) + \
                         m * torch.log(1 - w**2) - m * torch.log(1 - rho**2 - v**2) + torch.log(u)

            if accept:
                return w

def _sample_orthonormal_complement(v):
    """Samples a random vector orthonormal to v."""
    dimension = v.size(1)
    if dimension < 2:
        raise ValueError("Dimension must be greater than or equal to 2 for orthonormal complement.")

    random_direction = torch.randn_like(v)
    projection = torch.sum(random_direction * v, dim=1, keepdim=True) * v / torch.sum(v * v, dim=1, keepdim=True)
    orthogonal_vector = random_direction - projection
    return orthogonal_vector / torch.linalg.norm(orthogonal_vector, dim=1, keepdim=True)

def _sample_uniform_sphere(dimension: int, num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Samples uniformly from the unit hypersphere using Gaussian sampling."""
    samples = torch.randn(num_samples, dimension, device=device, dtype=dtype)
    return samples / torch.linalg.norm(samples, dim=1, keepdim=True)

if __name__ == '__main__':
    # Example usage:
    dimension = 3
    mu = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    kappa = 10.0
    num_samples = 5

    samples = sample_vmf(mu, kappa, num_samples)
    print("Samples from von Mises-Fisher distribution:")
    print(samples)

    # Verify that the samples are on the unit sphere (approximately)
    norms = torch.linalg.norm(samples, dim=1)
    print("\nNorms of the samples:")
    print(norms)