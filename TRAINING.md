# Training Configurations

Complete guide to training and evaluation commands for all experiments in the paper.

## Prerequisites

**Download required data files** from Google Drive and place them in the `data/` folder:
- [https://drive.google.com/drive/folders/19Ayyfoeddp2yR3cDXJELxYEZJ7xfV8mX?usp=sharing](https://drive.google.com/drive/folders/19Ayyfoeddp2yR3cDXJELxYEZJ7xfV8mX?usp=sharing)

This includes pre-computed files for GMM, DCT, vMF, PCA pruning axes, and other source distributions.

---

## Table of Contents

- [Baseline Models](#baseline-models)
- [Source Approximation Methods](#source-approximation-methods)
- [Directional Alignment Methods](#directional-alignment-methods)
- [Proposed Methods](#proposed-methods)
- [Evaluation](#evaluation)
- [Command-Line Arguments](#command-line-arguments)

---

## Baseline Models

### OT-CFM (Optimal Transport Conditional Flow Matching)
```bash
python train_cifar10.py \
    --model otcfm \
    --source gaussian \
    --batch_size 512
```

### I-CFM (Independent Conditional Flow Matching)
```bash
python train_cifar10.py \
    --model icfm \
    --source gaussian \
    --batch_size 512
```

### $\chi$-Sphere Source
```bash
python train_cifar10.py \
    --model otcfm \
    --source norm_gauss \
    --batch_size 512 \
    --subname x0_chisphere
```

---

## Source Approximation Methods

### DCT (Discrete Cosine Transform)

**DCT-Weak (Conservative masking):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source dct \
    --mask weak \
    --batch_size 512 \
    --x1_process=True \
    --subname weak
```

**DCT-Strong (Aggressive masking):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source dct \
    --mask strong \
    --batch_size 512 \
    --x1_process=True \
    --subname strong
```

### GMM (Gaussian Mixture Model)

**GMM-1 (Single Gaussian):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source gmm \
    --ncluster 1 \
    --batch_size 512
```

**GMM-2 (Two components):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source gmm \
    --ncluster 2 \
    --batch_size 512
```

**GMM-10 (Ten components):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source gmm \
    --ncluster 10 \
    --batch_size 512
```

### FFJORD (Neural ODE-based)

```bash
python train_cifar10.py \
    --model otcfm \
    --source ffjord \
    --batch_size 512
```

---

## Directional Alignment Methods

### Oracle-vMF (von Mises-Fisher with known directions)

**Note**: `vonmises` source requires `--vmf_all=True`. Each data sample is used as the mean direction.

**κ = 3000 (High concentration):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source vonmises \
    --vmf_all=True \
    --kappa 3000 \
    --batch_size 512 \
    --subname kappa3000 \
    --X0_ChiSphere=True 
```

**κ = 1500 (Medium concentration):**
```bash
python train_cifar10.py \
    --model otcfm \
    --source vonmises \
    --vmf_all=True \
    --kappa 1500 \
    --batch_size 512 \
    --subname kappa1500 \
    --X0_ChiSphere=True 
```

### Kmeans-vMF (Learned cluster directions)

**κ = 1500, K = 3:**
```bash
python train_cifar10.py \
    --model otcfm \
    --source sknn_vmf \
    --kappa 1500 \
    --ncluster 3 \
    --batch_size 512 \
    --subname kappa1500 \
    --X0_ChiSphere=True 
```

**κ = 300, K = 3:**
```bash
python train_cifar10.py \
    --model otcfm \
    --source sknn_vmf \
    --kappa 300 \
    --ncluster 3 \
    --batch_size 512 \
    --subname kappa300 \
    --X0_ChiSphere=True 
```

**κ = 50, K = 3:**
```bash
python train_cifar10.py \
    --model otcfm \
    --source sknn_vmf \
    --kappa 50 \
    --ncluster 3 \
    --batch_size 512 \
    --subname kappa50 \
    --X0_ChiSphere=True 
```

---

## Proposed Methods

### Norm Alignment

**OT-CFM + Norm Align:**
```bash
python train_cifar10.py \
    --model otcfm \
    --source gaussian \
    --X0_NormAlign=True \
    --batch_size 512 \
    --subname X0_NormAlign
```

**I-CFM + Norm Align:**
```bash
python train_cifar10.py \
    --model icfm \
    --source gaussian \
    --X0_NormAlign=True \
    --batch_size 512 \
    --subname X0_NormAlign
```

---

## Evaluation

### Standard FID Computation

**Single checkpoint:**
```bash
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm \
    --source gaussian \
    --start_step 80000 \
    --end_step 180000 \
    --step 10000 \
    --num_gen 50000 \
    --batch_size_fid 512
```

### Pruned Sampling (Post-hoc Improvement)

**With threshold 0.048:**
```bash
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm \
    --source pca_pruned \
    --threshold 0.048 \
    --start_step 80000 \
    --end_step 180000 \
    --step 10000 \
    --num_gen 50000 \
    --batch_size_fid 512
```

### Multi-GPU Evaluation

**Standard FID (distributed):**
```bash
bash scripts/fid_cifar10.sh
```

**Pruned FID (distributed):**
```bash
bash scripts/fid_cifar10_pruned.sh
```

---

## Command-Line Arguments

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `otcfm` | Flow matching model type (`otcfm`, `icfm`) |
| `--source` | `gaussian` | Source distribution type |
| `--batch_size` | `512` | Training batch size |
| `--lr` | `2e-4` | Learning rate |
| `--total_steps` | `200001` | Total training steps |
| `--save_step` | `10000` | Checkpoint save frequency |
| `--subname` | `""` | Experiment subdirectory name |

### Source Distribution Options

| Source Type | Description | Additional Args |
|-------------|-------------|----------------|
| `gaussian` | Standard Gaussian N(0, I) | - |
| `dct` | DCT-based frequency masking | `--mask weak/strong` |
| `gmm` | Gaussian Mixture Model | `--ncluster N` |
| `ffjord` | Neural ODE-based | - |
| `vonmises` | von Mises-Fisher (oracle, requires `--vmf_all=True`) | `--kappa K --vmf_all=True` |
| `sknn_vmf` | Spherical k-means + vMF | `--kappa K --ncluster N` |
| `pca_pruned` | PCA-based pruning (inference only) | `--threshold T` |

### Normalization Options

| Argument | Description |
|----------|-------------|
| `--X0_NormAlign` | Align source norm to target (recommended) |
| `--X1_NormAlign` | Align target norm to source |
| `--X0_Norm` | Normalize source to unit sphere |
| `--X1_Norm` | Normalize target to unit sphere |
| `--X0_ChiSphere` | Project source to chi-distributed sphere |
| `--X1_ChiSphere` | Project target to chi-distributed sphere |

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | - | Directory containing checkpoints |
| `--start_step` | - | First checkpoint step to evaluate |
| `--end_step` | - | Last checkpoint step to evaluate |
| `--step` | - | Step size between evaluations |
| `--num_gen` | `50000` | Number of samples to generate |
| `--batch_size_fid` | `512` | Batch size for FID computation |
| `--threshold` | - | Rejection threshold for pruned sampling |

### Clustering Arguments (for GMM/vMF sources)

| Argument | Default | Description |
|----------|---------|-------------|
| `--ncluster` | `3` | Number of clusters/components |
| `--kappa` | - | Concentration parameter for vMF |
| `--vmf_all` | `False` | Use all data for vMF (oracle mode) |

---

## Advanced: Preparing Source Distributions

### 1. Generate DCT Masks
```bash
cd 0_source_approx/DCT
python create_dct_masks.py \
    --output_dir ../../data \
    --mean_threshold 0.1 \
    --var_threshold_weak 2.0 \
    --var_threshold_strong 4.0
```

### 2. Fit GMM to CIFAR-10
```bash
cd 0_source_approx/GMM
python gmm_fit.py \
    --n_components 10 \
    --covariance_type full \
    --output_dir ../../data
```

### 3. Train FFJORD
```bash
cd 0_source_approx/FFJORD
python train_cnf.py \
    --data cifar10 \
    --dims 64,64,64 \
    --num_epochs 400 \
    --save ../../data/ffjord_ckpt.pth
```

### 4. Fit Spherical K-Means
```bash
cd 1_dir_align
python fit_spherical_kmeans.py \
    --n_clusters 3 \
    --augment_flip \
    --output_dir ../data
```

### 5. Compute PCA Pruning Axes
```bash
cd 2_pruning
python find_pruning_axis_with_pca.py \
    --dataset cifar10 \
    --rejection_threshold 0.01 \
    --pca_method incremental

# Output: ../data/pca_pruned_flip_under_1e-2.pt
```

---

## Example Workflows

### Complete Baseline Experiment
```bash
# 1. Train model
python train_cifar10.py --model otcfm --source gaussian --batch_size 512

# 2. Evaluate standard
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm --source gaussian \
    --start_step 80000 --end_step 180000 --step 10000 \
    --num_gen 50000

# 3. Evaluate with pruning (post-hoc)
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm --source pca_pruned --threshold 0.048 \
    --start_step 80000 --end_step 180000 --step 10000 \
    --num_gen 50000
```

### Reproduce Best Results
```bash
# Train with norm alignment
python train_cifar10.py \
    --model icfm \
    --source gaussian \
    --X0_NormAlign=True \
    --batch_size 512 \
    --subname X0_NormAlign

# Evaluate with pruned sampling
python compute_fid_cifar.py \
    --input_dir results/icfm_gaussian_X0_NormAlign \
    --model icfm --source pca_pruned --threshold 0.048 \
    --start_step 80000 --end_step 180000 --step 10000 \
    --num_gen 50000
# Expected: FID ≈ 3.64 @ NFE=100
```

---

## Notes

- **Batch Size**: Adjust based on your GPU memory (512 works on 40GB+ GPUs)
- **Multi-GPU**: Use `accelerate` for distributed training (see `accelerator/` configs)
- **Checkpoints**: Saved in `results/{model}_{source}_{subname}/`
- **Pruning**: Can be applied to ANY pre-trained model without retraining!
