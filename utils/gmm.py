import torch
__all__ = ["GMM_TORCH"]

class GMM_TORCH:
    def __init__(self, gmm, gmm_stats=None, stats_type='channel_wise', dtype=torch.float32, device="cpu"):
        """
        Convert a scikit-learn Gaussian Mixture Model (GMM) into a PyTorch-compatible format.

        Args:
            gmm (sklearn.mixture.GaussianMixture): Pre-trained GMM model.
            gmm_stats (ditionary): mean and standard deviation of GMM samples
            dtype (torch.dtype): Data type for PyTorch tensors.
            device (str or torch.device): Device to store tensors (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.dtype = dtype

        # Convert GMM parameters to PyTorch tensors
        self.weights = torch.tensor(gmm.weights_, dtype=dtype, device=device)
        self.means = torch.tensor(gmm.means_, dtype=dtype, device=device)
        self.covariances = torch.tensor(gmm.covariances_, dtype=dtype, device=device)
        self.n_cluster = self.means.shape[0]
        if gmm_stats is not None:
            print("Using Normalization Sampling")
            self.mu, self.std = gmm_stats['mean'].mean([2,3], keepdim=True).to(device), gmm_stats['std'].mean([2,3], keepdim=True).to(device)
            print(self.mu, self.std)
        else:
            print("Not Using Normalization Sampling")
            self.mu, self.std = 0, 1
        # Precompute Cholesky decomposition for faster sampling
        self.cholesky_factors = torch.linalg.cholesky(self.covariances)

    def sample(self, num_samples, return_cluster=False):
        """
        Generate samples from the Gaussian Mixture Model using PyTorch.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples of shape (num_samples, dim).
        """
        if return_cluster:
            x, cluster = self._torch_gmm_sample(num_samples, return_cluster=return_cluster)
        else:
            x = self._torch_gmm_sample(num_samples)
        x = x.reshape(-1, 3, 32, 32)
        if return_cluster:
            return (x-self.mu) / self.std, cluster
        else:
            return (x-self.mu) / self.std
    
    def sample_from_cluster(self, num_samples, cluster_idx):
        """
        특정 클러스터에서 num_samples 개수만큼 샘플을 생성
    
        Args:
            num_samples (int): 샘플 개수
            cluster_idx (int): 선택할 클러스터 인덱스 (0 ~ K-1)
    
        Returns:
            torch.Tensor: (num_samples, 3, 32, 32) 크기의 샘플
        """
        dim = self.means.shape[1]
        
        # 선택한 클러스터의 평균과 Cholesky 분해된 공분산
        mean = self.means[cluster_idx]
        cholesky = self.cholesky_factors[cluster_idx]
    
        # 표준 정규 분포에서 샘플 생성
        standard_samples = torch.randn(num_samples, dim, device=self.device)
    
        # 가우시안 변환: x = mean + L @ z
        if cholesky.ndim == 2:
            samples = mean + torch.mm(standard_samples, cholesky.T)
        else:
            samples = mean + torch.bmm(standard_samples.unsqueeze(1), cholesky.transpose(-2, -1)).squeeze(1)
            
        # 원래 이미지 크기로 reshape
        samples = samples.reshape(-1, 3, 32, 32)
        
        # 정규화 적용 후 반환
        return samples
        
    def _torch_gmm_sample(self, num_samples, return_cluster=False):
        """ 
        Private method for sampling from the GMM.
        
        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples of shape (num_samples, dim).
        """
        # Sample component indices based on mixture weights
        component_ids = torch.multinomial(self.weights, num_samples, replacement=True)

        # Generate standard normal samples
        dim = self.means.shape[1]
        standard_samples = torch.randn(num_samples, dim, device=self.device)

        # Select corresponding means and Cholesky factors
        chosen_means = self.means[component_ids]
        chosen_cholesky = self.cholesky_factors[component_ids]  # (num_samples, dim, dim)

        # Apply transformation: x = mean + L @ standard_samples
        samples = chosen_means + torch.bmm(chosen_cholesky, standard_samples.unsqueeze(-1)).squeeze(-1)

        if return_cluster:
            return samples, component_ids
        else:
            return samples

# class GMM_TORCH:
#     def __init__(self, gmm, gmm_stats, dtype=torch.float32, device="cpu"):
#         """
#         Convert a scikit-learn Gaussian Mixture Model (GMM) into a PyTorch-compatible format.

#         Args:
#             gmm (sklearn.mixture.GaussianMixture): Pre-trained GMM model.
#             dtype (torch.dtype): Data type for PyTorch tensors.
#             device (str or torch.device): Device to store tensors (e.g., 'cpu' or 'cuda').
#         """
#         self.device = device
#         self.dtype = dtype

#         # Convert GMM parameters to PyTorch tensors
#         self.weights = torch.tensor(gmm.weights_, dtype=dtype, device=device)
#         self.means = torch.tensor(gmm.means_, dtype=dtype, device=device)
#         self.covariances = torch.tensor(gmm.covariances_, dtype=dtype, device=device)
#         self.mu, self.std = gmm_stats['mean'].reshape(3, 32, 32).unsqueeze(0).mean([2,3], keepdim=True).to(device), gmm_stats['std'].reshape(3, 32, 32).unsqueeze(0).mean([2,3], keepdim=True).to(device)
#         # Precompute Cholesky decomposition for faster sampling
#         self.cholesky_factors = torch.linalg.cholesky(self.covariances)

#     def sample(self, num_samples):
#         """
#         Generate samples from the Gaussian Mixture Model using PyTorch.

#         Args:
#             num_samples (int): Number of samples to generate.

#         Returns:
#             torch.Tensor: Samples of shape (num_samples, dim).
#         """
#         x = self._torch_gmm_sample(num_samples)
#         x = x.reshape(-1, 3, 32, 32)
#         return (x-self.mu) / self.std

#     def _torch_gmm_sample(self, num_samples):
#         """ 
#         Private method for sampling from the GMM.
        
#         Args:
#             num_samples (int): Number of samples to generate.

#         Returns:
#             torch.Tensor: Samples of shape (num_samples, dim).
#         """
#         # Sample component indices based on mixture weights
#         component_ids = torch.multinomial(self.weights, num_samples, replacement=True)

#         # Generate standard normal samples
#         dim = self.means.shape[1]
#         standard_samples = torch.randn(num_samples, dim, device=self.device)

#         # Select corresponding means and Cholesky factors
#         chosen_means = self.means[component_ids]
#         chosen_cholesky = self.cholesky_factors[component_ids]  # (num_samples, dim, dim)

#         # Apply transformation: x = mean + L @ standard_samples
#         samples = chosen_means + torch.bmm(chosen_cholesky, standard_samples.unsqueeze(-1)).squeeze(-1)

#         return samples