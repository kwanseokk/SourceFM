# 2D Flow Matching Simulations

This directory contains 2D simulations that provide intuition for high-dimensional flow matching behavior under different source distributions.

## Overview

**Purpose**: Visualize and quantify the impact of different source distribution choices on flow matching in an interpretable 2D setting that mirrors high-dimensional phenomena.

**Key Insight**: 2D simulations reveal path entanglement, mode collapse, and omnidirectional learning effects that translate to high-dimensional image generation.

---

## Files

- **`run_whole_simulation.py`**: Main script running all 10 simulation scenarios
- **`2d_simulation.ipynb`**: Interactive notebook for exploring simulations
- **`aggregate_score.ipynb`**: Aggregates and visualizes metrics across runs
- **`ckpts/`**: Pre-trained base models at different iterations (20, 200, 6k, 8.5k, 10k)
- **`results/`**: Generated trajectories, visualizations, and metric files

---

## Quick Start

### Run All Simulations

```bash
cd simulations
python run_whole_simulation.py
```

**Runtime**: ~2-3 hours (10 scenarios × 10 runs × 100k iterations)

**Output**:
- Trajectory visualizations in `results/{scenario}/`
- Metrics in `results/metrics_results_{id}.npy`

### Interactive Exploration

```bash
jupyter notebook 2d_simulation.ipynb
```

---

## Simulation Scenarios

### Target Distribution

All simulations use a **3-mode circular target** distribution sampled on a chi-distributed circle:
- 3 directional clusters with ratios [5%, 30%, 65%]
- Radius follows χ²(625) distribution (~25 average)
- 1024 fixed target samples

### Source Distributions Tested

| ID | Scenario | Source Distribution | Purpose |
|----|----------|-------------------|---------|
| 0 | **OT-CFM** | Standard circle (χ²(3072)) | Baseline with optimal transport |
| 1 | **CFM** | Standard circle | Independent CFM baseline |
| 2 | **CFM-Rej** | Pruned circle (rejection threshold=0.98) | **Our method**: Inference pruning |
| 3 | **CFM-Iter20** | Density approximation (20 iter model output) | Weak density approximation |
| 4 | **CFM-Iter200** | Density approximation (200 iter model output) | Medium density approximation |
| 5 | **CFM-Iter6000** | Density approximation (6k iter model output) | Strong density approximation |
| 6 | **DirICFM** | Directional (3 modes, loose) | Directional alignment - loose |
| 7 | **DirPerfectPairing** | Directional (3 modes) + OT pairing | Directional + perfect OT |
| 8 | **DirICFM-Tight** | Directional (3 modes, tight threshold=0.98) | Directional alignment - tight |
| 9 | **CFM-Iter10000** | Density approximation (10k iter model output) | Near-perfect density approximation |

---

## Metrics Computed

Each simulation tracks 7 metrics:

1. **Average Min Distance**: Mean L2 distance from generated to nearest real sample
2. **% Bad Samples**: Percentage of generated samples >1.0 distance from data
3. **Wasserstein-2 Distance**: Optimal transport distance between distributions
4. **Coverage**: Fraction of real data near generated samples (ε=0.5)
5. **MMD Score**: Maximum Mean Discrepancy (kernel bandwidth σ=1.0)
6. **MDD**: Mode Distribution Divergence (KL divergence of mode assignments)
7. **Normalized Wasserstein**: Wasserstein distance normalized by mode distribution

---

## Key Findings from Simulations

### 1. Density Approximation Degrades Performance

**Scenarios 3-5, 9**: Using model outputs as source distributions

**Result**: ❌ Stronger approximation → worse performance

**Why**: Mode mismatch between training samples (discrete) and source (continuous approximation) confuses the model.

### 2. Directional Alignment Without Perfect Pairing Fails

**Scenarios 6, 8**: Directional source without optimal transport

**Result**: ❌ Path entanglement even with correct directions
- DirICFM (loose): Better than baseline but suboptimal
- DirICFM-Tight: Worse - over-concentration causes crossing paths

**Why**: Without perfect pairing, flows from different source modes can target the same data modes, causing interference.

### 3. Directional + Perfect Pairing Works

**Scenario 7**: Directional source WITH optimal transport pairing

**Result**: ✅ Best performance when source modes perfectly match target modes

**Limitation**: Requires knowing target modes a priori (not practical for real data).

### 4. Pruned Sampling is Robust

**Scenario 2**: Standard training + inference pruning

**Result**: ✅ Consistent improvement over baseline
- Avoids data-sparse regions during sampling
- No path entanglement (still uses full Gaussian for training)
- Works without knowing target structure

---

## Visualizations

Each scenario generates trajectory plots showing:
- **Black dots**: Source samples
- **Blue dots**: Target data
- **Red dots**: Generated samples
- **Light blue/red lines**: Trajectories (colored by final distance to data)

Example output: `results/OT-CFM/0/OT-CFM_10k.png`

---

## Reproducing Paper Figures

The 2D simulation figures in the paper correspond to:
- **Figure 2a**: CFM-Iter200 (scenario 4)
- **Figure 2b**: OT-CFM (scenario 0)
- **Figure 2c**: CFM-Iter10000 (scenario 9)
- **Figure 3a**: DirICFM (scenario 6)
- **Figure 3b**: DirICFM-Tight (scenario 8)
- **Figure 3c**: DirPerfectPairing (scenario 7)
- **Figure 4a**: CFM (scenario 1)
- **Figure 4b**: OT-CFM (scenario 0)
- **Figure 4c**: CFM-Rej (scenario 2)

---

## Notes

- **Dimension**: 2D (interpretable) but uses χ²(3072) to match high-D statistics
- **Training**: 100k iterations per run, 10 runs per scenario
- **Model**: Simple 3-layer MLP with time embedding
- **ODE Solver**: Dopri5 with adjoint sensitivity (atol=1e-4, rtol=1e-4)
- **Fixed Seed**: 124142 for reproducibility
