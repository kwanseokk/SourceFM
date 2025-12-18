import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils.extmath import randomized_svd
from safetensors.torch import load_file
from tqdm import tqdm
from glob import glob
from PIL import Image
import gc

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================
class FFHQDataset(Dataset):
    def __init__(self, root_dir='/data/dataset/ffhq/images1024x1024', image_size=256):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_paths = sorted(glob(os.path.join(self.root_dir, '**', '*.png'), recursive=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
def assign_to_centroids(features: torch.Tensor,
                        centroids: torch.Tensor,
                        metric: str = 'cosine',
                        device: str = 'cpu',
                        batch_size: int = 1024) -> torch.Tensor:
    """
    Assigns each feature vector in X to the closest centroid using batched processing to avoid OOM.

    Args:
        features (torch.Tensor): A tensor of shape (N, D) where N is the number of samples
                                 and D is the feature dimension.
        centroids (torch.Tensor): A tensor of shape (K, D) where K is the number of clusters.
        metric (str): The distance metric to use, either 'cosine' or 'euclidean'.
        device (str): The device ('cpu' or 'cuda') to perform computations on.
        batch_size (int): Number of features to process at once to manage memory usage.

    Returns:
        torch.Tensor: A 1D long tensor of shape (N,) containing cluster indices (0 to K-1)
                      for each feature vector.
    """
    centroids = centroids.to(device)
    n_samples = features.shape[0]
    assignments = torch.empty(n_samples, dtype=torch.long)
    
    if metric == 'cosine':
        # Normalize centroids once
        centroids_norm = F.normalize(centroids, p=2, dim=1)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_features = features[i:end_idx].to(device)
            
            # Normalize batch features
            features_norm = F.normalize(batch_features, p=2, dim=1)
            # Calculate cosine similarity as a matrix multiplication.
            similarities = torch.mm(features_norm, centroids_norm.t())  # Shape: (batch, K)
            # The assignment is the index of the centroid with the maximum similarity.
            batch_assignments = similarities.argmax(dim=1).cpu()
            assignments[i:end_idx] = batch_assignments
            
            # Clear GPU memory
            del batch_features, features_norm, similarities, batch_assignments
            torch.cuda.empty_cache()

    elif metric == 'euclidean':
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_features = features[i:end_idx].to(device)
            
            # Calculate pairwise Euclidean distances.
            distances = torch.cdist(batch_features, centroids, p=2)  # Shape: (batch, K)
            # The assignment is the index of the centroid with the minimum distance.
            batch_assignments = distances.argmin(dim=1).cpu()
            assignments[i:end_idx] = batch_assignments
            
            # Clear GPU memory
            del batch_features, distances, batch_assignments
            torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine' or 'euclidean'.")

    return assignments


def load_safetensors_dataset(load_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads image and label tensors from a directory of .safetensors files.

    Args:
        load_path (str): The path to the directory containing .safetensors files.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the concatenated
                                           images tensor and labels tensor.
    """
    image_list = []
    label_list = []
    print(f"Loading dataset from {load_path}...")
    for file_name in sorted(os.listdir(load_path)):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(load_path, file_name)
            try:
                data = load_file(file_path)
                image_list.append(data["images"])
                label_list.append(data["labels"])
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")

    if not image_list or not label_list:
        raise ValueError("No valid .safetensors files were found or loaded.")

    images = torch.cat(image_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return images, labels

def load_latent_dataset(load_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    .safetensors 파일 디렉토리에서 latent, latent_flip, label 텐서를 로드합니다.

    Args:
        load_path (str): .safetensors 파일이 포함된 디렉토리 경로.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            하나로 합쳐진 (latents, latents_flip, labels) 텐서 튜플.
    """
    latent_list = []
    latent_flip_list = []
    label_list = []
    
    print(f"Loading dataset from {load_path}...")
    for file_name in sorted(os.listdir(load_path)):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(load_path, file_name)
            try:
                data = load_file(file_path)
                latent_list.append(data["latents"])
                latent_flip_list.append(data["latents_flip"])
                label_list.append(data["labels"])
                
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")
    if not latent_list or not label_list:
        raise ValueError("유효한 .safetensors 파일을 찾을 수 없거나 로드에 실패했습니다.")
    # 각 리스트의 텐서들을 하나로 합칩니다.
    latents = torch.cat(latent_list, dim=0)
    latents_flip = torch.cat(latent_flip_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return latents, latents_flip, labels

def load_cifar10_augmented() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the CIFAR-10 training dataset and augments it with horizontal flips.
    Images are flattened and L2-normalized.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the augmented data tensor
                                           of shape (100000, 3072) and the corresponding
                                           labels tensor of shape (100000,).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                                  # Convert to [0, 1] tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Scale to [-1, 1]
    ])

    full_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)

    # Separate images and labels into tensors.
    images = torch.stack([img for img, _ in full_dataset])  # Shape: [50000, 3, 32, 32]
    labels = torch.tensor([lbl for _, lbl in full_dataset])   # Shape: [50000]

    # Create horizontally flipped versions of the images.
    flipped_images = torch.flip(images, dims=[3])  # Flip along the width dimension

    # Flatten both original and flipped images.
    num_samples = images.size(0)
    flat_images = images.view(num_samples, -1)              # Shape: [50000, 3072]
    flat_flipped_images = flipped_images.view(num_samples, -1) # Shape: [50000, 3072]

    # Combine original and flipped data.
    combined_data = torch.cat([flat_images, flat_flipped_images], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)

    # L2-normalize the combined dataset.
    normalized_data = F.normalize(combined_data, p=2, dim=1)

    return normalized_data, combined_labels


def incremental_pca(data_loader: DataLoader, n_components: int, device: str = 'cuda', batch_size_pca: int = None):
    """
    Performs memory-efficient incremental PCA using sklearn's IncrementalPCA.
    
    Args:
        data_loader (DataLoader): DataLoader yielding batches of shape (B, D).
        n_components (int): The number of principal components to compute.
        device (str): The device to perform computations on.
        batch_size_pca (int): Batch size for incremental PCA fitting.
    
    Returns:
        dict: A dictionary containing 'components_', 'mean_', and 'explained_variance_'.
    """
    print("Starting incremental PCA...")
    
    # Get data dimension from first batch
    first_batch = next(iter(data_loader))
    if isinstance(first_batch, (list, tuple)):
        first_batch = first_batch[0]
    n_features = first_batch.shape[1]
    
    # Use smaller batch size for PCA if not specified
    if batch_size_pca is None:
        batch_size_pca = min(1000, n_components * 2)
    
    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size_pca)
    
    print(f"Fitting incremental PCA with batch_size={batch_size_pca}...")
    for batch_data in tqdm(data_loader, desc="PCA fitting"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        
        # Move to CPU for sklearn compatibility
        batch_cpu = batch.cpu().numpy()
        ipca.partial_fit(batch_cpu)
        
        # Clear GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    pca_result = {
        'components_': torch.tensor(ipca.components_, dtype=torch.float32),
        'mean_': torch.tensor(ipca.mean_, dtype=torch.float32),
        'explained_variance_': torch.tensor(ipca.explained_variance_, dtype=torch.float32),
    }
    print("Incremental PCA complete.")
    return pca_result


def estimate_memory_usage(n_samples, n_features, dtype=torch.float32):
    """
    Estimate memory usage for fast PCA.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        dtype: Data type
    
    Returns:
        dict: Memory usage estimates in GB
    """
    bytes_per_element = 4 if dtype == torch.float32 else 8
    
    # Original data: N × D
    data_memory = n_samples * n_features * bytes_per_element
    
    # Centered data: N × D (temporary)
    centered_memory = n_samples * n_features * bytes_per_element
    
    # SVD components: U(N×min(N,D)) + S(min(N,D)) + V(D×min(N,D))
    min_dim = min(n_samples, n_features)
    u_memory = n_samples * min_dim * bytes_per_element
    s_memory = min_dim * bytes_per_element
    v_memory = n_features * min_dim * bytes_per_element
    svd_memory = u_memory + s_memory + v_memory
    
    # Peak memory (data + centered + SVD components)
    peak_memory = data_memory + centered_memory + svd_memory
    
    return {
        'data_gb': data_memory / (1024**3),
        'centered_gb': centered_memory / (1024**3), 
        'svd_components_gb': svd_memory / (1024**3),
        'peak_gb': peak_memory / (1024**3),
        'recommended_gpu_memory_gb': peak_memory * 1.5 / (1024**3)  # 50% buffer
    }


def fast_pca_pytorch(X, n_components=256, device=None):
    """
    Fast PyTorch PCA using SVD - mathematically equivalent to standard PCA.
    Includes memory usage estimation and warnings for large datasets.
    
    Args:
        X (torch.Tensor): Input data tensor of shape (N, D).
        n_components (int): Number of principal components to compute.
        device (str): Device to use. If None, uses CUDA if available.
    
    Returns:
        tuple: (components, singular_values, mean) where components has shape (D, n_components)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_samples, n_features = X.shape
    print(f"Running fast PyTorch PCA on {device} with {n_samples:,} samples, {n_features:,} features")
    
    # Estimate memory usage
    memory_info = estimate_memory_usage(n_samples, n_features, X.dtype)
    print(f"📊 Memory estimation:")
    print(f"  - Data size: {memory_info['data_gb']:.2f} GB")
    print(f"  - Peak memory usage: {memory_info['peak_gb']:.2f} GB") 
    print(f"  - Recommended GPU memory: {memory_info['recommended_gpu_memory_gb']:.2f} GB")
    
    # Check if this is likely to cause OOM
    if memory_info['peak_gb'] > 20:  # > 20GB is risky
        print("⚠️  WARNING: Very large memory usage detected!")
        print("   Consider using 'incremental' or 'randomized' method instead.")
        
        # Ask for confirmation (in interactive mode)
        import sys
        if sys.stdin.isatty():  # Interactive terminal
            response = input("Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                raise RuntimeError("Operation cancelled due to memory concerns.")
    
    # Check available GPU memory
    if device == 'cuda' and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_available_gb = gpu_memory_gb - torch.cuda.memory_reserved(0) / (1024**3)
        print(f"🎮 GPU memory: {gpu_available_gb:.1f} GB available / {gpu_memory_gb:.1f} GB total")
        
        if memory_info['peak_gb'] > gpu_available_gb * 0.8:  # Use 80% as safety margin
            print("❌ Insufficient GPU memory! Falling back to CPU...")
            device = 'cpu'
    
    X = X.to(device)
    
    # Center the data (same as all other methods)
    print("Computing mean...")
    X_mean = X.mean(dim=0, keepdim=True)
    print("Centering data...")
    X_centered = X - X_mean
    
    try:
        # Standard PCA: perform SVD on X_centered directly
        print("Performing SVD on centered data...")
        U, S, V = torch.svd(X_centered)  # U: (N, min(N,D)), S: (min(N,D),), V: (D, min(N,D))
        
        # Select top n_components
        components = V[:, :n_components]  # Shape: (D, n_components)
        singular_values = S[:n_components]  # Shape: (n_components,)
        
        print("✅ Fast PyTorch PCA completed successfully!")
        return components, singular_values, X_mean.squeeze(0)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"💥 GPU Out of Memory Error: {e}")
            print("🔄 Try using:")
            print("   --pca_method incremental  (most memory efficient)")
            print("   --pca_method randomized   (good balance)")
            raise RuntimeError("GPU out of memory. Use incremental or randomized PCA method.")
        else:
            raise


def fast_pca_pytorch_batched(data_loader: DataLoader, n_components: int, device: str = 'cuda'):
    """
    Fast PyTorch PCA for large datasets that don't fit in memory.
    Uses batched approach with SVD.
    
    Args:
        data_loader (DataLoader): DataLoader yielding batches of shape (B, D).
        n_components (int): The number of principal components to compute.
        device (str): The device to perform computations on.
    
    Returns:
        dict: A dictionary containing 'components_', 'mean_', and 'explained_variance_'.
    """
    print("Starting fast PyTorch batched PCA...")
    
    # Collect all data (if memory allows)
    print("Loading all data into memory...")
    data_list = []
    n_samples = 0
    
    for batch_data in tqdm(data_loader, desc="Loading batches"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        data_list.append(batch.cpu())  # Keep on CPU initially
        n_samples += batch.shape[0]
    
    # Concatenate all data
    all_data = torch.cat(data_list, dim=0)
    del data_list  # Free memory
    
    print(f"Data shape: {all_data.shape}")
    
    try:
        # Try to use fast PyTorch PCA
        components, singular_values, mean = fast_pca_pytorch(
            all_data, n_components=n_components, device=device
        )
        
        # Calculate explained variance
        explained_variance = (singular_values ** 2) / (n_samples - 1)
        
        pca_result = {
            'components_': components.t().cpu(),  # Transpose to match sklearn format: (n_components, D)
            'mean_': mean.cpu(),
            'explained_variance_': explained_variance.cpu(),
        }
        
        print("Fast PyTorch PCA complete.")
        return pca_result
        
    except RuntimeError as e:
        print(f"Fast PyTorch PCA failed (likely out of memory): {e}")
        print("Falling back to randomized PCA...")
        
        # Fallback to randomized PCA
        return randomized_pca_sklearn_fallback(all_data, n_components, device)


def randomized_pca_sklearn_fallback(data: torch.Tensor, n_components: int, device: str):
    """
    Fallback randomized PCA using sklearn when PyTorch SVD fails.
    """
    print("Using sklearn randomized SVD fallback...")
    
    # Convert to numpy for sklearn
    data_np = data.cpu().numpy()
    n_samples = data_np.shape[0]
    
    # Compute mean
    mean = np.mean(data_np, axis=0)
    centered_data = data_np - mean
    
    # Use randomized SVD
    U, s, Vt = randomized_svd(
        centered_data, 
        n_components=n_components, 
        n_iter=4,
        random_state=42
    )
    
    # Convert back to torch tensors
    pca_result = {
        'components_': torch.tensor(Vt, dtype=torch.float32),
        'mean_': torch.tensor(mean, dtype=torch.float32),
        'explained_variance_': torch.tensor(s**2 / (n_samples - 1), dtype=torch.float32),
    }
    
    return pca_result


def randomized_pca_torch(data_loader: DataLoader, n_components: int, device: str = 'cuda', 
                        n_iter: int = 4, random_state: int = None):
    """
    Performs randomized PCA for very large datasets using randomized SVD.
    Now with fallback to fast PyTorch PCA when possible.
    
    Args:
        data_loader (DataLoader): DataLoader yielding batches of shape (B, D).
        n_components (int): The number of principal components to compute.
        device (str): The device to perform computations on.
        n_iter (int): Number of iterations for randomized SVD.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        dict: A dictionary containing 'components_', 'mean_', and 'explained_variance_'.
    """
    print("Starting randomized PCA...")
    
    # First try the fast PyTorch batched approach
    try:
        return fast_pca_pytorch_batched(data_loader, n_components, device)
    except Exception as e:
        print(f"Fast PyTorch batched PCA failed: {e}")
        print("Falling back to original randomized approach...")
    
    # Original implementation as fallback
    n_samples = 0
    mean = 0
    
    print("Computing mean...")
    for batch_data in tqdm(data_loader, desc="Mean computation"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        batch = batch.cpu()  # Keep on CPU to save GPU memory
        n = batch.shape[0]
        mean += batch.sum(dim=0)
        n_samples += n
    mean /= n_samples
    
    # Collect all centered data for randomized SVD
    print("Collecting centered data...")
    centered_data_list = []
    
    for batch_data in tqdm(data_loader, desc="Data centering"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        batch = batch.cpu()
        centered = batch - mean
        centered_data_list.append(centered)
    
    # Combine all data
    all_data = torch.cat(centered_data_list, dim=0).numpy()
    
    # Perform randomized SVD
    print(f"Performing randomized SVD with {n_components} components...")
    U, s, Vt = randomized_svd(
        all_data, 
        n_components=n_components, 
        n_iter=n_iter,
        random_state=random_state
    )
    
    # Convert back to torch tensors
    pca_result = {
        'components_': torch.tensor(Vt, dtype=torch.float32),
        'mean_': mean,
        'explained_variance_': torch.tensor(s**2 / (n_samples - 1), dtype=torch.float32),
    }
    
    print("Randomized PCA complete.")
    return pca_result


def batched_pca(data_loader: DataLoader, n_components: int, device: str = 'cuda'):
    """
    Improved batched PCA with better memory management and optional GPU acceleration.
    
    Args:
        data_loader (DataLoader): DataLoader yielding batches of shape (B, D).
        n_components (int): The number of principal components to compute.
        device (str): The device to perform computations on.

    Returns:
        dict: A dictionary containing 'components_', 'mean_', and 'explained_variance_'.
    """
    n_samples = 0
    mean = 0
    
    print("Starting improved batched PCA...")
    
    # First pass: compute the global mean with better memory management
    print("Pass 1/3: Computing mean...")
    for batch_data in tqdm(data_loader, desc="Mean computation"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data 
        batch = batch.to(device, non_blocking=True)
        n = batch.shape[0]
        mean += batch.sum(dim=0)
        n_samples += n
        
        # Clear batch from GPU memory immediately
        del batch
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    mean /= n_samples

    # Second pass: compute running covariance matrix more efficiently
    print("Pass 2/3: Computing covariance matrix...")
    n_features = mean.shape[0]
    
    # Check if covariance matrix is too large for memory
    cov_size_gb = (n_features ** 2 * 8) / (1024 ** 3)  # 8 bytes for float64
    print(f"Covariance matrix would require {cov_size_gb:.1f} GB")
    
    if cov_size_gb > 8:  # Reduced threshold from 20GB to 8GB for safety
        print("Covariance matrix too large - switching to incremental PCA...")
        # Use incremental PCA which is more memory efficient
        return incremental_pca(data_loader, n_components, device, batch_size_pca=1000)
    
    # Use double precision for better numerical stability
    cov = torch.zeros((n_features, n_features), device=device, dtype=torch.float64)
    
    for batch_data in tqdm(data_loader, desc="Covariance computation"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        batch = batch.to(device, dtype=torch.float64, non_blocking=True)
        centered = batch - mean.double()
        
        # More memory-efficient covariance update
        cov.addmm_(centered.T, centered)
        
        # Clear GPU memory
        del batch, centered
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    cov /= (n_samples - 1)

    # Third pass: eigenvalue decomposition with better algorithms
    print("Pass 3/3: Performing eigenvalue decomposition...")
    
    # Use symeig for symmetric matrices (more stable)
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)
    except RuntimeError as e:
        print(f"GPU eigendecomposition failed: {e}")
        print("Falling back to CPU...")
        cov_cpu = cov.cpu()
        eigvals, eigvecs = torch.linalg.eigh(cov_cpu)
        eigvals = eigvals.to(device)
        eigvecs = eigvecs.to(device)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_idx][:n_components]
    eigvecs = eigvecs[:, sorted_idx][:, :n_components]

    pca_result = {
        'components_': eigvecs.T.float(),         # Shape: (n_components, D)
        'mean_': mean.float(),                    # Shape: (D,)
        'explained_variance_': eigvals.float(),   # Shape: (n_components,)
    }
    
    # Clean up
    del cov
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print("Improved batched PCA complete.")
    return pca_result


def batched_pca_randomized_svd(data_loader: DataLoader, n_components: int, device: str, mean: torch.Tensor, n_samples: int):
    """
    Memory-efficient PCA using randomized SVD when covariance matrix is too large.
    Uses streaming approach to minimize memory usage.
    """
    from sklearn.utils.extmath import randomized_svd
    import numpy as np
    import tempfile
    import os
    
    print("Using randomized SVD for memory-efficient PCA...")
    
    # Much smaller chunk size to prevent OOM
    max_samples_per_chunk = 2000  # Reduced from 8000
    temp_files = []  # Store chunk data in temporary files
    chunk_info = []  # Store metadata about chunks
    
    print("Processing data in streaming chunks...")
    current_chunk = []
    current_size = 0
    chunk_idx = 0
    
    for batch_data in tqdm(data_loader, desc="Processing batches"):
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        
        # Process in very small sub-batches to minimize peak memory
        batch_size = batch.shape[0]
        sub_batch_size = min(512, batch_size)  # Process 512 samples at a time
        
        for i in range(0, batch_size, sub_batch_size):
            end_idx = min(i + sub_batch_size, batch_size)
            sub_batch = batch[i:end_idx].to(device, non_blocking=True)
            
            # Center and convert to float32 for memory efficiency
            centered = (sub_batch - mean).cpu().numpy().astype(np.float32)
            current_chunk.append(centered)
            current_size += centered.shape[0]
            
            del sub_batch
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Save chunk to temporary file when it gets large enough
            if current_size >= max_samples_per_chunk:
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_chunk_{chunk_idx}.npy')
                chunk_matrix = np.vstack(current_chunk)
                np.save(temp_file.name, chunk_matrix)
                
                temp_files.append(temp_file.name)
                chunk_info.append({
                    'file': temp_file.name,
                    'shape': chunk_matrix.shape,
                    'samples': chunk_matrix.shape[0]
                })
                
                # Clear memory
                del current_chunk, chunk_matrix
                current_chunk = []
                current_size = 0
                chunk_idx += 1
                
                # Force garbage collection
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        del batch
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Handle remaining data
    if current_chunk:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_chunk_{chunk_idx}.npy')
        chunk_matrix = np.vstack(current_chunk)
        np.save(temp_file.name, chunk_matrix)
        
        temp_files.append(temp_file.name)
        chunk_info.append({
            'file': temp_file.name,
            'shape': chunk_matrix.shape,
            'samples': chunk_matrix.shape[0]
        })
        
        del current_chunk, chunk_matrix
        gc.collect()
    
    print(f"Created {len(chunk_info)} temporary chunks")
    
    try:
        if len(chunk_info) == 1:
            # Single chunk case
            print(f"Loading single chunk for SVD...")
            data_matrix = np.load(chunk_info[0]['file'])
            print(f"SVD on matrix of shape {data_matrix.shape}")
            
            n_comp = min(n_components, data_matrix.shape[0], data_matrix.shape[1])
            U, s, Vt = randomized_svd(data_matrix, n_components=n_comp, n_iter=5, random_state=42)
            
            components = Vt[:n_components]
            explained_variance = (s[:n_components]**2) / (n_samples - 1)
            
        else:
            # Multiple chunks - use incremental approach
            print("Processing multiple chunks with incremental SVD...")
            
            # Start with first chunk
            print(f"Loading first chunk: {chunk_info[0]['shape']}")
            first_chunk = np.load(chunk_info[0]['file'])
            
            n_comp = min(n_components, first_chunk.shape[0], first_chunk.shape[1])
            U, s, Vt = randomized_svd(first_chunk, n_components=n_comp, n_iter=3, random_state=42)
            
            del first_chunk
            gc.collect()
            
            # For simplicity, use first chunk's components
            # This is an approximation but memory-safe
            components = Vt[:n_components]
            explained_variance = (s[:n_components]**2) / (n_samples - 1)
            
            print("Note: Using first chunk components for memory efficiency")
        
        pca_result = {
            'components_': torch.tensor(components, dtype=torch.float32),
            'mean_': mean.cpu(),
            'explained_variance_': torch.tensor(explained_variance, dtype=torch.float32),
        }
        
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")
    
    print("Randomized SVD PCA complete.")
    return pca_result

def find_reject_directions(centroids: torch.Tensor,
                            data: torch.Tensor,
                            threshold: float,
                            cache_path: str,
                            batch_size: int = 10_000,
                            device: str = 'cuda',
                            data_loader: DataLoader = None) -> list[int]:
    """
    Finds centroids whose maximum cosine similarity to any data point is below a threshold.
    This function caches the max similarity computation to a file to avoid re-computation.
    Assumes both centroids and data are already L2-normalized.

    Args:
        centroids (torch.Tensor): Normalized centroids tensor of shape (K, D).
        data (torch.Tensor): Normalized data tensor of shape (N, D), or None if using data_loader.
        threshold (float): The similarity threshold for rejection.
        cache_path (str): Path to the file for saving/loading max similarity values.
        batch_size (int): The batch size for processing data to save memory.
        device (str): The device to perform computations on.
        data_loader (DataLoader): Optional DataLoader for memory-efficient processing.

    Returns:
        list[int]: A list of indices of the rejected centroids.
    """
    # --- Step 1: Check for a pre-computed cache file ---
    if os.path.exists(cache_path):
        print(f"Loading pre-computed max similarities from '{cache_path}'...")
        max_similarities = torch.load(cache_path).to(device)
    else:
        # --- Step 2: If no cache, compute max similarities for ALL centroids ---
        print(f"No cache file found. Computing max similarities for all centroids...")
        centroids = centroids.to(device)
        num_centroids = centroids.shape[0]
        
        # Initialize a tensor to store the max similarity for each centroid.
        max_similarities = torch.full((num_centroids,), -float('inf'), device=device)

        # Iterate through each centroid to find its true max similarity across the entire dataset.
        for i in tqdm(range(num_centroids), desc="Computing Max Similarity"):
            c = centroids[i:i+1]  # Shape: (1, D)
            
            if data_loader is not None:
                # Use DataLoader for memory-efficient processing
                for batch_data in data_loader:
                    if isinstance(batch_data, (list, tuple)):
                        data_batch = batch_data[0].to(device)
                    else:
                        data_batch = batch_data.to(device)
                    
                    sim = torch.matmul(c, data_batch.T)  # Shape: (1, B)
                    batch_max = sim.max().item()

                    # Update the max similarity found so far for the current centroid.
                    if batch_max > max_similarities[i]:
                        max_similarities[i] = batch_max
                    
                    del data_batch, sim
                    if device == 'cuda':
                        torch.cuda.empty_cache()
            else:
                # Use data tensor directly
                for start in range(0, data.shape[0], batch_size):
                    end = start + batch_size
                    data_batch = data[start:end].to(device)  # Shape: (B, D)

                    sim = torch.matmul(c, data_batch.T)  # Shape: (1, B)
                    batch_max = sim.max().item()

                    # Update the max similarity found so far for the current centroid.
                    if batch_max > max_similarities[i]:
                        max_similarities[i] = batch_max
                    
                    del data_batch, sim
                    if device == 'cuda':
                        torch.cuda.empty_cache()
        
        # --- Step 3: Save the computed values to the cache file ---
        print(f"Saving computed max similarities to '{cache_path}'...")
        # Save to CPU to ensure compatibility across different devices/environments.
        torch.save(max_similarities.cpu(), cache_path)

    # --- Step 4: Filter indices based on the threshold ---
    print(f"Finding directions with max similarity <= {threshold}")
    # Find all indices where the condition is met.
    reject_indices_tensor = (max_similarities <= threshold).nonzero(as_tuple=True)[0]
    
    return reject_indices_tensor.cpu().tolist()

def choose_pca_method(method: str, data_loader: DataLoader, n_components: int, device: str, **kwargs):
    """
    Choose and execute the appropriate PCA method based on user selection.
    
    Args:
        method (str): PCA method to use ('batched', 'incremental', 'randomized', 'fast')
        data_loader (DataLoader): DataLoader for the dataset
        n_components (int): Number of principal components
        device (str): Computing device
        **kwargs: Additional arguments for specific PCA methods
    
    Returns:
        dict: PCA result with components_, mean_, and explained_variance_
    """
    print(f"Using {method} PCA method...")
    
    if method == 'incremental':
        return incremental_pca(
            data_loader, 
            n_components, 
            device, 
            batch_size_pca=kwargs.get('pca_batch_size')
        )
    elif method == 'randomized':
        return randomized_pca_torch(
            data_loader, 
            n_components, 
            device,
            n_iter=kwargs.get('n_iter', 4),
            random_state=kwargs.get('random_state', 42)
        )
    elif method == 'fast':
        return fast_pca_pytorch_batched(data_loader, n_components, device)
    elif method == 'batched':
        return batched_pca(data_loader, n_components, device)
    else:
        raise ValueError(f"Unknown PCA method: {method}")


def compare_pca_methods(data: torch.Tensor, n_components: int = 10, device: str = 'cuda'):
    """
    Compare different PCA methods to ensure they produce mathematically equivalent results.
    
    Args:
        data (torch.Tensor): Test data of shape (N, D)
        n_components (int): Number of components to compare
        device (str): Device to run computations on
    
    Returns:
        dict: Comparison results and similarity metrics
    """
    print("Comparing PCA methods for mathematical equivalence...")
    
    # Create a simple dataloader for the test data
    from torch.utils.data import TensorDataset
    test_dataset = TensorDataset(data)
    test_loader = DataLoader(test_dataset, batch_size=min(1000, data.shape[0]), shuffle=False)
    
    results = {}
    
    try:
        # Test sklearn PCA (reference)
        print("1. Testing sklearn PCA (reference)...")
        from sklearn.decomposition import PCA as SklearnPCA
        data_np = data.cpu().numpy()
        sklearn_pca = SklearnPCA(n_components=n_components)
        sklearn_pca.fit(data_np)
        results['sklearn'] = {
            'components': torch.tensor(sklearn_pca.components_, dtype=torch.float32),
            'mean': torch.tensor(sklearn_pca.mean_, dtype=torch.float32),
            'explained_variance': torch.tensor(sklearn_pca.explained_variance_, dtype=torch.float32)
        }
        
        # Test fast PyTorch PCA
        print("2. Testing fast PyTorch PCA...")
        fast_components, fast_singular_values, fast_mean = fast_pca_pytorch(data, n_components, device)
        fast_explained_var = (fast_singular_values ** 2) / (data.shape[0] - 1)
        results['fast'] = {
            'components': fast_components.t().cpu(),  # Transpose to match sklearn format
            'mean': fast_mean.cpu(),
            'explained_variance': fast_explained_var.cpu()
        }
        
        # Test batched PCA
        print("3. Testing batched PCA...")
        batched_result = batched_pca(test_loader, n_components, device)
        results['batched'] = {
            'components': batched_result['components_'],
            'mean': batched_result['mean_'].cpu(),
            'explained_variance': batched_result['explained_variance_'].cpu()
        }
        
        # Compare results
        print("\n=== COMPARISON RESULTS ===")
        for method in ['fast', 'batched']:
            if method in results:
                # Compare components (absolute value since direction can be flipped)
                component_similarity = torch.mean(torch.abs(torch.abs(results['sklearn']['components']) - 
                                                           torch.abs(results[method]['components']))).item()
                
                # Compare means
                mean_diff = torch.mean(torch.abs(results['sklearn']['mean'] - results[method]['mean'])).item()
                
                # Compare explained variance
                var_diff = torch.mean(torch.abs(results['sklearn']['explained_variance'] - 
                                              results[method]['explained_variance'])).item()
                
                print(f"{method.upper()} vs sklearn:")
                print(f"  Component difference (avg): {component_similarity:.2e}")
                print(f"  Mean difference (avg): {mean_diff:.2e}")
                print(f"  Explained variance diff (avg): {var_diff:.2e}")
                print(f"  {'✅ EQUIVALENT' if component_similarity < 1e-4 else '❌ DIFFERENT'}")
                print()
        
        return results
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        return None


def to_scientific_str(value):
  formatted_str = f"{value:.0e}"
  return formatted_str.replace('e-0', 'e-').replace('e+0', 'e+')
# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="PCA-based Clustering and Analysis")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet64', 'ffhq', 'sdv2_imagenet32', 'imagenet256'],
                        help='Dataset to use.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the computations on.')
    parser.add_argument('--rejection_threshold', type=float, default=0.01,
                        help='Cosine similarity threshold for rejecting a direction.')
    # ====================   NEW ARGUMENT   ====================
    parser.add_argument('--cache_path_root', type=str, default='../data',
                        help='Path to save/load the cache files.')
    parser.add_argument('--pca_method', type=str, default='incremental', 
                        choices=['batched', 'incremental', 'randomized', 'fast'],
                        help='PCA method to use: batched (original), incremental (memory efficient), randomized (sklearn-based), or fast (PyTorch SVD).')
    parser.add_argument('--pca_batch_size', type=int, default=None,
                        help='Batch size for incremental PCA. If None, will be set automatically.')
    parser.add_argument('--test_equivalence', action='store_true',
                        help='Test mathematical equivalence of PCA methods on a small dataset before running main analysis.')
    # =========================================================

    args = parser.parse_args()
    print(f"Configuration: {args}")
    
    # Test mathematical equivalence if requested
    if args.test_equivalence:
        print("🧪 Testing PCA method equivalence...")
        # Generate synthetic test data
        torch.manual_seed(42)
        test_data = torch.randn(500, 100)  # 500 samples, 100 features
        test_data = F.normalize(test_data, p=2, dim=1)  # L2 normalize like in the main code
        
        comparison_results = compare_pca_methods(test_data, n_components=10, device=args.device)
        
        if comparison_results:
            print("✅ Equivalence test completed. Check results above.")
        else:
            print("❌ Equivalence test failed.")
        print("="*60)
    
    # Create cache directory
    cache_dir = os.path.join(args.cache_path_root, args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_path = os.path.join(cache_dir, 'max_similarities.pt')
    pca_cache_path = os.path.join(cache_dir, f'pca_components_{args.pca_method}.pt')
    # --- Load and Preprocess Data ---
    if args.dataset == 'cifar10':
        data, labels = load_cifar10_augmented()
        dim = data.shape[1]
        
        # Check if PCA components are cached
        if os.path.exists(pca_cache_path):
            print(f"Loading cached PCA components from '{pca_cache_path}'...")
            components = torch.load(pca_cache_path)
        else:
            print("Fitting PCA on CIFAR-10 data...")
            pca = PCA(n_components=dim)
            pca.fit(data.numpy())
            
            components = torch.tensor(pca.components_, dtype=torch.float32)
            
            # Save PCA components to cache
            print(f"Saving PCA components to cache '{pca_cache_path}'...")
            torch.save(components, pca_cache_path)
        
        batch_size = 50_000
    elif args.dataset == 'imagenet64':
        images, labels = load_safetensors_dataset("/data/dataset/imagenet/1K_imagenet64/train")
        dim = images.shape[1] * images.shape[2] * images.shape[3] # 64*64*3
        
        # Flatten and normalize the images.
        data = images.view(-1, dim)
        data = F.normalize(data.to(torch.float32), p=2, dim=1)

        # Check if PCA components are cached
        if os.path.exists(pca_cache_path):
            print(f"Loading cached PCA components from '{pca_cache_path}'...")
            components = torch.load(pca_cache_path)
        else:
            # For large datasets, use batched PCA.
            dataloader = DataLoader(
                TensorDataset(data),
                batch_size=50_000,
                shuffle=False,
                num_workers=4,
            )
            pca_result = choose_pca_method(args.pca_method, dataloader, n_components=dim, device=args.device, pca_batch_size=args.pca_batch_size)
            components = pca_result['components_'].cpu()
            
            # Save PCA components to cache
            print(f"Saving PCA components to cache '{pca_cache_path}'...")
            torch.save(components, pca_cache_path)
        
        batch_size = 50_000
    elif args.dataset == 'sdv2_imagenet32':
        images, images_flip, labels = load_latent_dataset("/data/dataset/imagenet/official_sdv2_imagenet_latent")
        dim = images.shape[1] * images.shape[2] * images.shape[3] # 32 * 32 * 4
        # Combine original and flipped images.
        # data = torch.cat([images, images_flip], dim=0)
        data = images
        # Flatten and normalize the images.
        data = data.view(-1, dim)
        data = F.normalize(data.to(torch.float32), p=2, dim=1)
        
        # Check if PCA components are cached
        if os.path.exists(pca_cache_path):
            print(f"Loading cached PCA components from '{pca_cache_path}'...")
            components = torch.load(pca_cache_path)
        else:
            # For large datasets, use batched PCA.
            dataloader = DataLoader(
                TensorDataset(data),
                batch_size=50_000,
                shuffle=False,
                num_workers=4,
            )
            pca_result = choose_pca_method(args.pca_method, dataloader, n_components=dim, device=args.device, pca_batch_size=args.pca_batch_size)
            components = pca_result['components_'].cpu()
            
            # Save PCA components to cache
            print(f"Saving PCA components to cache '{pca_cache_path}'...")
            torch.save(components, pca_cache_path)
        
        batch_size = 50_000
    elif args.dataset == 'imagenet256':
        from torchvision.datasets import ImageFolder
        dataset_path = os.path.join("/data/dataset/imagenet/1K_dataset", 'train')
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])
        
        # Create dataset
        dataset = ImageFolder(dataset_path, transform=transform_train)
        dim = 3 * 256 * 256  # 256*256*3
        
        # Create a custom dataset that flattens and normalizes on the fly
        class FlattenedImageNet256Dataset(Dataset):
            def __init__(self, base_dataset, dim):
                self.base_dataset = base_dataset
                self.dim = dim
                
            def __len__(self):
                return len(self.base_dataset)
                
            def __getitem__(self, idx):
                img, _ = self.base_dataset[idx]  # ImageFolder returns (image, label)
                # Flatten and normalize
                flat_img = img.view(-1)  # Shape: (dim,)
                normalized_img = F.normalize(flat_img.unsqueeze(0), p=2, dim=1).squeeze(0)
                return normalized_img
        
        flattened_dataset = FlattenedImageNet256Dataset(dataset, dim)
        
        # Check if PCA components are cached
        if os.path.exists(pca_cache_path):
            print(f"Loading cached PCA components from '{pca_cache_path}'...")
            components = torch.load(pca_cache_path)
            
            # For rejection analysis, create a more memory-efficient approach
            print("Setting up data loader for rejection analysis...")
            dataloader = DataLoader(
                flattened_dataset,
                batch_size=256,  # Reduced batch size for memory efficiency
                shuffle=False,
                num_workers=2,   # Reduced workers to save memory
                pin_memory=False,  # Disable pin_memory to save GPU memory
                prefetch_factor=1  # Reduce prefetching
            )
            # Don't load all data at once for large datasets
            # The rejection analysis will process in batches
            data = None  # Will be processed batch by batch
            
        else:
            # For large datasets, use batched PCA
            dataloader = DataLoader(
                flattened_dataset,
                batch_size=256,  # Reduced from 512 for memory efficiency
                shuffle=False,
                num_workers=2,   # Reduced from 4
                pin_memory=False  # Disabled for memory efficiency
            )
            pca_result = choose_pca_method(args.pca_method, dataloader, n_components=dim, device=args.device, pca_batch_size=args.pca_batch_size)
            components = pca_result['components_'].cpu()
            
            # Save PCA components to cache
            print(f"Saving PCA components to cache '{pca_cache_path}'...")
            torch.save(components, pca_cache_path)
            
            # Don't load all data into memory at once
            data = None  # Will be processed batch by batch
        batch_size = 512
    elif args.dataset == 'ffhq':
        # Create dataset without loading all images into memory
        ffhq_dataset = FFHQDataset()
        dim = 3 * 256 * 256  # Assuming images are resized to 256x256
        
        # Create a custom dataset that flattens and normalizes on the fly
        class FlattenedFFHQDataset(Dataset):
            def __init__(self, base_dataset, dim):
                self.base_dataset = base_dataset
                self.dim = dim
                
            def __len__(self):
                return len(self.base_dataset)
                
            def __getitem__(self, idx):
                img = self.base_dataset[idx]
                # Flatten and normalize
                flat_img = img.view(-1)  # Shape: (dim,)
                normalized_img = F.normalize(flat_img.unsqueeze(0), p=2, dim=1).squeeze(0)
                return normalized_img
        
        flattened_dataset = FlattenedFFHQDataset(ffhq_dataset, dim)
        
        # Create dataloader with memory-efficient batching
        dataloader = DataLoader(
            flattened_dataset,
            batch_size=1024,  # Smaller batch size for memory efficiency
            shuffle=False,
            num_workers=4,
            pin_memory=True if args.device == 'cuda' else False,
            persistent_workers=True
        )
        
        # Check if PCA components are cached
        if os.path.exists(pca_cache_path):
            print(f"Loading cached PCA components from '{pca_cache_path}'...")
            components = torch.load(pca_cache_path)
            
            # Still need to load data for rejection analysis
            print("Loading flattened dataset for rejection analysis...")
            data_list = []
            for batch in tqdm(dataloader, desc="Loading data"):
                data_list.append(batch)
            data = torch.cat(data_list, dim=0)
            del data_list  # Free memory
        else:
            pca_result = choose_pca_method(args.pca_method, dataloader, n_components=dim, device=args.device, pca_batch_size=args.pca_batch_size)
            components = pca_result['components_'].cpu()
            
            # Save PCA components to cache
            print(f"Saving PCA components to cache '{pca_cache_path}'...")
            torch.save(components, pca_cache_path)
            
            # Get flattened data for later use
            print("Loading flattened dataset for rejection analysis...")
            data_list = []
            for batch in tqdm(dataloader, desc="Loading data"):
                data_list.append(batch)
            data = torch.cat(data_list, dim=0)
            del data_list  # Free memory
        
        batch_size = 1024
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not implemented.")
        
    # --- Create Centroids from PCA Components ---
    positive_centroids = components
    negative_centroids = -1 * components
    centroids = torch.cat([positive_centroids, negative_centroids], dim=0)
    print(f"Created {centroids.shape[0]} centroids from {components.shape[0]} principal components.")

    # Handle memory-efficient rejection analysis
    if data is None:
        # Use dataloader for memory-efficient processing
        reject_direction_indices = find_reject_directions(
            centroids=centroids,
            data=None,
            threshold=args.rejection_threshold,
            cache_path=cache_path,  # Pass the cache path
            batch_size=batch_size,
            device=args.device,
            data_loader=dataloader
        )
    else:
        reject_direction_indices = find_reject_directions(
            centroids=centroids,
            data=data,
            threshold=args.rejection_threshold,
            cache_path=cache_path,  # Pass the cache path
            batch_size=batch_size,
            device=args.device
        )
    
    print("Filtering out rejected centroids...")
    all_indices = set(range(centroids.shape[0]))
    rejected_indices_set = set(reject_direction_indices)
    accepted_indices = sorted(list(all_indices - rejected_indices_set))
    
    accepted_centroids = centroids[accepted_indices]
    num_accepted = accepted_centroids.shape[0]
    print(f"Accepted {num_accepted} centroids out of {centroids.shape[0]}.")

    if num_accepted == 0:
        print("Warning: All centroids were rejected. No results will be saved.")
    else:
        print("Re-assigning data to accepted centroids...")
        if data is None:
            # For memory efficiency, compute assignments in batches using dataloader
            print("Computing assignments in batches for memory efficiency...")
            all_assignments = []
            for batch_data in tqdm(dataloader, desc="Computing assignments"):
                if isinstance(batch_data, (list, tuple)):
                    batch = batch_data[0]
                else:
                    batch = batch_data
                batch_assignments = assign_to_centroids(batch, accepted_centroids, device=args.device, batch_size=512)
                all_assignments.append(batch_assignments)
            new_assignments = torch.cat(all_assignments, dim=0)
            del all_assignments  # Free memory
        else:
            new_assignments = assign_to_centroids(data, accepted_centroids, device=args.device)
        
        unique_clusters, counts = torch.unique(new_assignments, return_counts=True)
        sample_ratio = counts.float() / counts.sum()
        print("Calculated new sample ratios for accepted clusters.")
        # --- Save Final Results ---
        output_dir = '../data'
        # Use consistent naming: pca_pruned_flip_under_1e-2.pt for CIFAR-10
        if args.dataset == 'cifar10':
            output_filename = f'pca_pruned_flip_under_{to_scientific_str(args.rejection_threshold)}.pt'
        else:
            output_filename = f'{args.dataset}_pca_pruned_flip_under_{to_scientific_str(args.rejection_threshold)}.pt'
        output_path = os.path.join(output_dir, output_filename)
        
        # results_to_save = {
        #     'centroids': accepted_centroids,
        #     'assignments': new_assignments,
        #     'sample_ratio': sample_ratio
        # }
        torch.save(accepted_centroids, output_path)
        print(f"Successfully saved final results to '{output_path}'")

    print("\n--- Results ---")
    print(f"Total number of rejected directions: {len(reject_direction_indices)}")

if __name__ == '__main__':
    main()