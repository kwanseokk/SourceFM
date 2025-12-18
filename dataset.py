import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import transforms
import random
from safetensors.torch import load_file
from glob import glob
import random
import os

class LDM_IMAGENET(Dataset):
    def __init__(self, path='/data/dataset/imagenet/sdv3_imagenet64_latent/', flip=True, normalize=True, cache_size=1):
        """
        Custom Dataset for loading ImageNet latent representations from safetensors files with caching and shape validation.
        
        Args:
            path (str): Directory path containing .safetensors files.
            flip (bool): If True, randomly select original or flipped latents.
            normalize (bool): If True, normalize latents using mu and std.
            cache_size (int): Number of safetensors files to cache in memory.
        """
        self.files = sorted(glob(os.path.join(path, "*.safetensors")))
        
        self.flip = flip
        self.normalize = normalize
        self.cache_size = cache_size
        
        # Count total number of samples and validate shapes
        self.total_samples = 0
        self.file_sample_counts = []
        for file in self.files:
            data = load_file(file)
            latents_shape = data['latents'].shape[0]
            labels_shape = data['labels'].shape[0]
            if latents_shape != labels_shape:
                print(f"Warning: File {file} has mismatch: latents {latents_shape}, labels {labels_shape}")
            self.file_sample_counts.append(latents_shape)
            self.total_samples += latents_shape
        
        try:
            self.stats = torch.load(os.path.join(path, "latents_stats.pt"))
        except:
            self.stats = self.calculate_stats()
        if not self.files:
            raise ValueError(f"No .safetensors files found in {path}")
        # Initialize cache
        self.cache = {}  # {file_path: data}
        self.cache_order = []  # To track order for LRU eviction
        
    def calculate_stats(self):
        latents = []
        for file in self.files:
            data = load_file(file)
            latents.append(data['latents'])
            latents.append(data['latents_flip'])
        latents = torch.cat(latents, dim=0)
        ch_mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        ch_std = latents.std(dim=[0, 2, 3], keepdim=True)
        total_mean = latents.mean()
        total_std = latents.std()
        self.stats = {'ch_mean': ch_mean, 'ch_std': ch_std, 'total_mean': total_mean, 'total_std': total_std}
        torch.save(self.stats, os.path.join(self.path, "latents_stats.pt"))
        return
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.total_samples
    
    def _load_file_with_cache(self, file_path):
        """
        Load a safetensors file, using cache if available.
        """
        if file_path in self.cache:
            return self.cache[file_path]
        
        # Load file
        data = load_file(file_path)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest file (LRU)
            oldest_file = self.cache_order.pop(0)
            del self.cache[oldest_file]
        
        self.cache[file_path] = data
        self.cache_order.append(file_path)
        
        return data
    
    def __getitem__(self, idx):
        """
        Get a single sample (latent, label) by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (latent, label)
                - latent (torch.Tensor): Latent representation ([C, H, W]).
                - label (torch.Tensor): Class label.
        """
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.total_samples}")
        
        # Find the file containing the idx-th sample
        file_idx = 0
        cumulative_samples = 0
        for i, count in enumerate(self.file_sample_counts):
            if idx < cumulative_samples + count:
                file_idx = i
                local_idx = idx - cumulative_samples
                break
            cumulative_samples += count
        else:
            raise IndexError(f"Index {idx} could not be mapped to any file")
        
        # Load the safetensors file with caching
        file_path = self.files[file_idx]
        data = self._load_file_with_cache(file_path)
        
        # Debug: Check shapes before accessing
        latents_shape = data['latents'].shape[0]
        labels_shape = data['labels'].shape[0]
        if local_idx >= latents_shape or local_idx >= labels_shape:
            raise IndexError(
                f"File {file_path}: local_idx {local_idx} is out of bounds "
                f"(latents shape[0]={latents_shape}, labels shape[0]={labels_shape})"
            )
        
        # Get latents and label
        latents = data['latents'][local_idx]  # [C, H, W]
        label = data['labels'][local_idx]     # Scalar or [1]
        
        # Handle flip
        if self.flip and random.random() < 0.5:
            if local_idx >= data['latents_flip'].shape[0]:
                raise IndexError(
                    f"File {file_path}: local_idx {local_idx} is out of bounds "
                    f"for latents_flip shape[0]={data['latents_flip'].shape[0]}"
                )
            latents = data['latents_flip'][local_idx]  # Use flipped latent
        
        # Handle normalization
        if self.normalize:
            latents = latents / self.stats["total_std"]
        return latents, label

class LDM_CIFAR10(Dataset):
    def __init__(self, path='data/sdv3_cifar10.safetensors', flip=True, normalize=True):
        safetensors = load_file(path)
        self.latents = safetensors['latents']
        self.latents_flip = safetensors['latents_flip']
        self.labels = safetensors['labels']
        self.mu = safetensors['mu']
        self.std = safetensors['std']

        self.flip = flip
        self.normalize = normalize
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.flip:
            image = self.latents[idx] if np.random.uniform(0, 1) > 0.5 else self.latents_flip[idx]
        else:
            image = self.latents[idx]
        if self.normalize:
            image = image / self.std
        labels = self.labels[idx]
        return image, labels

class CIFAR10WithClusters(Dataset):
    def __init__(self, root, train=True, transform=None, download=False, clusters_path='./data/clusters_dp.npy'):
        self.cifar10_dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        if clusters_path.endswith('.npy'):
            self.clusters = np.load(clusters_path)
        else:
            self.clusters = torch.load(clusters_path)
        if len(self.cifar10_dataset) != len(self.clusters):
            raise ValueError(
                f"The length of CIFAR-10 dataset ({len(self.cifar10_dataset)}) "
                f"must be the same as the length of clusters_dp.npy ({len(self.clusters)})"
            )

    def __len__(self):
        return len(self.cifar10_dataset)

    def __getitem__(self, idx):
        image, _ = self.cifar10_dataset[idx]
        cluster = self.clusters[idx]
        return image, cluster
    
class CIFAR10WithClustersFlip(Dataset):
    def __init__(self, root='data', train=True, cluster_path='data/spherical_kmeans3_flip.pt'):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )
        
        # Load cluster assignments
        cluster_data = torch.load(cluster_path)
        self.cluster_assignments = cluster_data['assignments']  # Tensor of shape [100000]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.train = train
        self.num_images = len(self.dataset)  # should be 50000

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        # Decide randomly whether to flip
        do_flip = random.random() < 0.5
        if do_flip:
            image = transforms.functional.hflip(image)
            cluster_idx = idx + self.num_images  # use flipped cluster assignment
        else:
            cluster_idx = idx  # use original cluster assignment

        image = self.transform(image)
        cluster_number = self.cluster_assignments[cluster_idx]

        return image, cluster_number.item()
