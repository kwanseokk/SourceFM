# from utils.sample_vmf import sample_vmf
from utils.sample_vmf2 import vMF
import torch
import numpy as np
from torch.nn import functional as F

class VMF:
    
    def __init__(self, params, normalize=False):
        self.params = params
        self.normalize = normalize
    
    # def sample(self, clusters):
    #     N = clusters.size(0)
    #     embedding_dim = 3072
    #     sampled_vectors = torch.zeros((N, embedding_dim), dtype=torch.float32, device=clusters.device)
        
    #     unique_clusters, counts = torch.unique(clusters, return_counts=True)
    #     kappa = self.params['kappa']
        
    #     for cluster in unique_clusters:
    #         mu = self.params[cluster.item()]
    #         indices = (clusters == cluster).nonzero(as_tuple=True)[0]
    #         sampled = sample_vmf(mu, kappa, num_samples=counts[cluster]).to(clusters.device)
    #         sampled_vectors[indices] = - sampled
            
    #     return sampled_vectors
    
    def sample(self, clusters):
        N = clusters.size(0)
        embedding_dim = 3072
        vmf = vMF(embedding_dim)
        sampled_vectors = torch.zeros((N, embedding_dim), dtype=torch.float32, device=clusters.device)
        
        unique_clusters, counts = torch.unique(clusters, return_counts=True)
        kappa = self.params['kappa']
        # kappa = torch.tensor(800.0).to(clusters.device)
        
        i = 0
        for cluster in unique_clusters:
            mu = self.params[cluster.item()]
            vmf.set_params(mu, kappa)
            indices = (clusters == cluster).nonzero(as_tuple=True)[0]
            sampled = vmf.sample(counts[i].item()).to(clusters.device)
            sampled_vectors[indices] = sampled
            i += 1
            
        if self.normalize:
            mean = torch.tensor([-0.0002, -0.0005, -0.0018], dtype=torch.float32, device=clusters.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            std = torch.tensor([0.0179, 0.0179, 0.0181], dtype=torch.float32, device=clusters.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            sampled_vectors = (sampled_vectors.view(-1, 3, 32, 32) - mean) / std
        else:
            sampled_vectors = sampled_vectors.view(-1, 3, 32, 32)
            
        return sampled_vectors
    
    def sample_vmf(self, x):
        embedding_dim = 3072
        # kappa = torch.tensor(1200.0).to(x.device)
        kappa = self.params['kappa']
        vmf = vMF(embedding_dim)
        
        x = x.view(-1, embedding_dim)
        x = F.normalize(x, dim=1)
        
        batch_samples = vmf.sample_batch(x, kappa)
        batch_samples = batch_samples.view(-1, 3, 32, 32)
        
        if self.normalize:
            mean = torch.tensor([-0.0002, -0.0005, -0.0015], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            std = torch.tensor([0.0180, 0.0179, 0.0181], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            batch_samples = (batch_samples - mean) / std
        
        return batch_samples
    
    def sample_vmf_mukappa(self, mu, kappa):
        embedding_dim = 3072
        vmf = vMF(embedding_dim)
        
        mu = mu.view(-1, embedding_dim)
        mu = F.normalize(mu, dim=1)
        
        batch_samples = vmf.sample_batch_mukappa(mu, kappa)
        batch_samples = batch_samples.view(-1, 3, 32, 32)
        
        if self.normalize:
            mean = torch.tensor([-0.0007, -0.0014, -0.0044], dtype=torch.float32, device=mu.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            std = torch.tensor([0.0172, 0.0172, 0.0185], dtype=torch.float32, device=mu.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            batch_samples = (batch_samples - mean) / std
        
        return batch_samples