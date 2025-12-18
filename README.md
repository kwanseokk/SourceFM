<!-- # Is There a Better Source Distribution than Gaussian?

**Exploring Source Distributions for Image Flow Matching**

[![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/forum?id=sev0GtV1fc&referrer=%5BAuthor%20Console%5D)

Official implementation of our TMLR submission investigating optimal source distribution choices for flow matching in generative modeling. -->

# Is There a Better Source Distribution than Gaussian?: Exploring Source Distributions for Image Flow Matching

![TMLR 2025](https://img.shields.io/badge/TMLR-2025-green)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/forum?id=sev0GtV1fc&referrer=%5BAuthor%20Console%5D)

[Junho Lee](mailto:joon2003@snu.ac.kr)\*,  [Kwanseok Kim](mailto:kjvd1009@snu.ac.kr)\*,  [Joonseok Lee](mailto:joonseok@snu.ac.kr)†

Seoul National University, Seoul, Korea  
\* Equal contribution  
† Corresponding author 

Official implementation of our TMLR submission investigating optimal source distribution choices for flow matching in generative modeling.

## TL;DR

We systematically explore alternatives to Gaussian source distributions for flow matching and discover:

- **Density approximation** (DCT, GMM, FFJORD) paradoxically degrades performance when approminated more
- **Directional alignment** (vMF distributions) suffers from path entanglement when tighter source is used
- ✅ **Gaussian's omnidirectional coverage** ensures robust learning across all directions
- ✅ **Our Solutions**:
  - **Norm Alignment** (training): Match source/target norms to reduce scale mismatch
  - **Pruned Sampling** (inference): Reject data-sparse initializations (post-hoc, no retraining!)

**Best Results on CIFAR-10 (FID ↓):**
- OT-CFM: 4.40 → **3.88** (Norm Align + Pruned)
- I-CFM: 4.35 → **3.64** (Norm Align + Pruned)

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd SourceFM

# Install dependencies
pip install torch torchvision torchdyn tqdm absl-py joblib lpips timm accelerate
```

### Download Required Data Files

Download pre-computed source distribution files (GMM, DCT, vMF, PCA pruning axes, etc.) and place them in the `data/` folder:

**Google Drive**: [https://drive.google.com/drive/folders/19Ayyfoeddp2yR3cDXJELxYEZJ7xfV8mX?usp=sharing](https://drive.google.com/drive/folders/19Ayyfoeddp2yR3cDXJELxYEZJ7xfV8mX?usp=sharing)

```bash
# After downloading, your data folder should contain:
data/
├── pca_pruned_flip_under_1e-2.pt
├── spherical_kmeans3_flip.pt
├── cifar10_gmm*_pca3072_full.pkl
├── noise_dctmask6_*.pt
└── ... (other source distribution files)
```

---

## Quick Start

### Training

**Baseline (Standard Gaussian):**
```bash
python train_cifar10.py --model otcfm --source gaussian --batch_size 512
```

**Our Method (Norm Alignment):**
```bash
python train_cifar10.py --model otcfm --source gaussian --X0_NormAlign=True --batch_size 512 --subname X0_NormAlign
```

### Evaluation with Pruned Sampling

**Standard Evaluation:**
```bash
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm --source gaussian \
    --start_step 80000 --end_step 180000 --step 10000 \
    --num_gen 50000 --batch_size_fid 512
```

**Pruned Sampling (Post-hoc Improvement):**
```bash
python compute_fid_cifar.py \
    --input_dir results/otcfm_gaussian \
    --model otcfm --source pca_pruned \
    --threshold 0.048 \
    --start_step 80000 --end_step 180000 --step 10000 \
    --num_gen 50000 --batch_size_fid 512
```

**📖 See [TRAINING.md](TRAINING.md) for complete training configurations and all experimental settings.**

---

## 2D Simulations

We introduce a novel 2D simulation framework that captures high-dimensional geometric properties in an interpretable setting. This enables direct visualization and analysis of flow matching learning dynamics during training.

**Key Contributions**:
- 2D environment that mirrors high-dimensional phenomena (path entanglement, mode discrepancy, omnidirectional learning)
- Direct visualization of flow trajectories and their evolution
- Quantitative metrics: Avg. Min. Distance, Failure Rate, Normalized Wasserstein. etc.

**Results**: Simulations reveal that density approximation risks mode discrepancy, directional alignment causes path entanglement when tighter, and Gaussian's broad coverage ensures stable learning across all regions.

**📖 See [simulations/](simulations/) for detailed documentation and code.**

---

## Experiments

We tested three hypotheses:

### 1. Source Density Approximation → [Details](0_source_approx/)

**Hypothesis**: Approximating the data distribution as a source should improve performance.

**Methods Tested**:
- **[DCT-based masking](0_source_approx/DCT/)**: Frequency-domain structured noise
- **[Gaussian Mixture Models](0_source_approx/GMM/)**: Multi-modal flexible distributions
- **[FFJORD](0_source_approx/FFJORD/)**: Neural ODE-based learned distributions

**Result**: ❌ Stronger approximation → **worse** performance (mode discrepancy problem)

### 2. Directional Alignment → [Details](1_dir_align/)

**Hypothesis**: Aligning source directions with data clusters improves flow paths.

**Methods Tested**:
- **[Spherical k-means + vMF](1_dir_align/fit_spherical_kmeans.py)**: Directionally-concentrated distributions
- Various concentration parameters (κ = 50, 300, 1500, 3000)

**Result**: ❌ Over-concentration causes **path entanglement** without perfect pairing.

### 3. Norm Alignment (Our Method - Training)

**Hypothesis**: Misalignment in average norms between source and target wastes model capacity.

**Method**:
- Match the average L2 norm of source distribution to target data
- Reduces computational resources needed to learn scale transformation
- Training-time modification: `--X0_NormAlign=True`

**Result**: ✅ Improves high-NFE performance

### 4. Pruned Sampling (Our Method - Inference) → [Details](2_pruning/)

**Hypothesis**: Train with full Gaussian, prune data-sparse regions at inference.

**Method**:
- Use PCA to identify directions with no nearby data
- Apply **pruned sampling** during inference only
- Can be applied **post-hoc** to any pre-trained model!

**Result**: ✅ Consistent improvements, across all NFEs

**Combined (Norm Align + Pruned)**: ✅ Best overall results across all NFE settings

---

## Repository Structure

```
SourceFM_CR/
├── README.md                     # This file
├── TRAINING.md                   # 📖 All training configs & arguments
├── train_cifar10.py              # Main training script
├── compute_fid_cifar.py          # FID evaluation with pruning support
├── setup_source.py               # Source distribution implementations
│
├── 0_source_approx/              # Density approximation experiments
│   ├── DCT/                      # → Frequency-domain masking
│   ├── GMM/                      # → Gaussian Mixture Models
│   └── FFJORD/                   # → Neural ODE-based sources
│
├── 1_dir_align/                  # Directional alignment experiments
│   ├── fit_spherical_kmeans.py  # → Spherical clustering + vMF
│   └── lab.ipynb                 # → Experimental notebook
│
├── 2_pruning/                    # Pruned sampling (our method)
│   ├── find_pruning_axis_with_pca.py  # → PCA-based pruning axis computation
│   └── README.md                 # → Detailed pruning documentation
│
├── simulations/                  # 2D flow matching simulations
│   ├── run_whole_simulation.py  # → All simulation scenarios
│   ├── 2d_simulation.ipynb      # → Interactive notebook
│   └── README.md                 # → Simulation documentation
│
├── torchcfm/                     # Flow matching library
├── utils/                        # Source distribution utilities
├── scripts/                      # Training/evaluation scripts
└── accelerator/                  # Multi-GPU configurations
```

**📖 See [TRAINING.md](TRAINING.md) for all training commands. Each directory contains detailed READMEs.**

---

## Results

### CIFAR-10 (FID ↓, NFE = Number of Function Evaluations)

| Method | NFE=5 | NFE=10 | NFE=20 | NFE=100 |
|--------|-------|--------|--------|---------|
| **OT-CFM Baseline** | 19.80 | 10.95 | 7.41 | 4.40 |
| + Pruned Sampling | **17.24** | **9.17** | **6.34** | 4.15 |
| + Norm Align + Pruned | 21.52 | 9.55 | 5.85 | **3.88** |
| **I-CFM Baseline** | 34.49 | 13.30 | 7.75 | 4.35 |
| + Pruned Sampling | **28.78** | **10.37** | **6.40** | 3.97 |
| + Norm Align + Pruned | 48.10 | 16.58 | 7.04 | **3.64** |

### ImageNet64 (FID ↓)

| Method | NFE=5 | NFE=10 | NFE=20 | NFE=100 |
|--------|-------|--------|--------|---------|
| OT-CFM Baseline | 53.95 | 18.84 | 10.62 | 9.10 |
| + Pruned Sampling | **49.56** | **16.70** | **9.54** | **8.78** |

---

## Key Insights

1. **Mode Discrepancy Problem**: Density approximation creates mode mismatch between training samples and source distribution, confusing the model.

2. **Path Entanglement**: Directional alignment concentrates flows, causing paths entangle. Only Oracle-vMF wins.

3. **Omnidirectional Learning**: Gaussian's broad coverage ensures all directions receive training signal, even if data-sparse.

4. **Norm Misalignment Costs**: Standard Gaussian has unit average norm, while CIFAR-10 images have ~55. This scale mismatch wastes model capacity during training.

5. **Two Solutions**:
   - **Norm Alignment** (training): Resolves scale mismatch 
   - **Pruned Sampling** (inference): Avoids data-sparse regions 
   - **Combined**: Best results across all NFE settings

---

## Citation

```bibtex
@article{sourcefm2024,
  title={Is There a Better Source Distribution than Gaussian? Exploring Source Distributions for Image Flow Matching},
  author={Junho Lee, Kwanseok Kim, Joonseok Lee},
  journal={Under review as submission to TMLR},
  year={2025}
}
```

---

## Acknowledgments

- Built on [TorchCFM](https://github.com/atong01/conditional-flow-matching)
- UNet architecture adapted from prior work

---

## Additional Documentation

- **[TRAINING.md](TRAINING.md)** - Complete training configurations and all experimental command-line arguments
- [0_source_approx/](0_source_approx/) - Density approximation methods (DCT, GMM, FFJORD)
- [1_dir_align/](1_dir_align/) - Directional alignment methods (Spherical k-means + vMF)
- [2_pruning/](2_pruning/) - Pruned sampling implementation
- [simulations/](simulations/) - 2D flow matching simulations and visualizations
- [scripts/](scripts/) - Shell scripts for distributed training/evaluation
- [utils/](utils/) - Source distribution implementations

For questions or issues, please open a GitHub issue.
