# Free-form Jacobian of Reversible Dynamics (FFJORD)

This directory contains code for training and using FFJORD as a source distribution for flow matching models.

FFJORD is a continuous normalizing flow model that learns flexible distributions through neural ODEs. In this project, we train FFJORD on CIFAR-10 and use it as an alternative source distribution instead of standard Gaussian noise.

## Overview

**Original Paper**:
> Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." _International Conference on Learning Representations_ (2019).
> [[arxiv]](https://arxiv.org/abs/1810.01367) [[bibtex]](http://www.cs.toronto.edu/~rtqichen/bibtex/ffjord.bib)

**Motivation**: Standard Gaussian noise may not be the optimal source distribution for flow matching. FFJORD learns a flexible, data-dependent source distribution by training a continuous normalizing flow model on real data. This learned distribution can potentially provide a better starting point for flow matching.

**Key Idea**:
1. Train FFJORD to model the data distribution using neural ODEs
2. Use the trained FFJORD model to sample from a learned source distribution
3. Use these samples as x₀ in flow matching instead of Gaussian noise

## Files

### Core Scripts

- **`train_cnf.py`**: Main training script for FFJORD
- **`train_misc.py`**: Utility functions for training (regularization, model building, etc.)

### Library Code

- **`lib/`**: Complete FFJORD implementation
  - `layers/`: Neural network layers including CNF (Continuous Normalizing Flow)
  - `odenvp.py`: Multiscale FFJORD architecture
  - `multiscale_parallel.py`: Parallel multiscale architecture
  - `utils.py`: General utilities

## Pre-trained Checkpoint

A pre-trained FFJORD checkpoint is available at `../../data/ffjord_ckpt.pth`.

**Training Configuration** (400 epochs):
```
Data: CIFAR-10
Architecture: Multiscale ODENVP
  - Dims: 64,64,64
  - Strides: 1,1,1,1
  - Num blocks: 2
  - Layer type: concat
  - Rademacher: True
```

This checkpoint can be loaded and used as a source distribution for flow matching models.

## Prerequisites

```bash
pip install torch torchvision torchdiffeq
```

The most important dependency is `torchdiffeq` for solving neural ODEs:
```bash
pip install git+https://github.com/rtqichen/torchdiffeq.git
```

## Usage

### 1. Training FFJORD (Optional)

If you want to train your own FFJORD model:

```bash
cd source_approx/FFJORD
python train_cnf.py \
    --data cifar10 \
    --dims 64,64,64 \
    --strides 1,1,1,1 \
    --num_blocks 2 \
    --layer_type concat \
    --multiscale True \
    --rademacher True \
    --num_epochs 400 \
    --batch_size 256 \
    --save experiments/ffjord_run
```

**Key Arguments**:
- `--data`: Dataset (mnist, svhn, cifar10, lsun_church)
- `--dims`: Hidden dimensions for the neural ODE network
- `--strides`: Strides for convolutional layers
- `--num_blocks`: Number of stacked CNF blocks
- `--layer_type`: Type of layer (concat, ignore, squash, etc.)
- `--multiscale`: Use multiscale architecture (recommended)
- `--rademacher`: Use Rademacher estimator for trace (faster)
- `--solver`: ODE solver (dopri5, bdf, rk4, etc.)

### 2. Resume Training from Checkpoint

```bash
python train_cnf.py \
    --resume ../../data/ffjord_ckpt.pth \
    --num_epochs 500 \
    --begin_epoch 401
```

### 3. Using FFJORD as a Source Distribution

The trained FFJORD model can be used as a source distribution via the `utils/ffjord.py` wrapper:

```python
from utils.ffjord import FFJORD

# Initialize FFJORD with pre-trained checkpoint
ffjord = FFJORD(device='cuda', ckpt_path='data/ffjord_ckpt.pth')

# Sample from the learned distribution
x0 = ffjord.sample(batch_size)  # Returns (B, 3, 32, 32) in range [-1, 1]
```

The `FFJORD` class provides:
- Automatic model architecture creation based on checkpoint config
- Efficient sampling in evaluation mode
- Proper scaling to [-1, 1] range for flow matching

## How FFJORD Works

### Continuous Normalizing Flows

FFJORD models data distributions using neural ODEs:

```
dz/dt = f(z(t), t; θ)
```

Where:
- `z(t)` is the state at time t
- `f` is a neural network (ODEnet)
- Starting from simple noise z(0) ~ N(0, I), we can generate data x = z(1)

### Training Objective

FFJORD is trained to maximize the log-likelihood of data:

```
log p(x) = log p(z(0)) - ∫₀¹ Tr(∂f/∂z) dt
```

The trace term (divergence) is computed efficiently using:
- **Hutchinson's trace estimator** with Rademacher random vectors
- No need to compute full Jacobian matrix

## Files Generated During Training

When training, the script creates:
```
experiments/cnf/
├── checkpt.pth                 # Latest checkpoint
├── logs                        # Training logs
└── figs/                       # Sample visualizations (if enabled)
```

Checkpoints contain:
- `state_dict`: Model weights
- `optim_state_dict`: Optimizer state
- `args`: Training configuration

## Integration with Flow Matching

FFJORD can be used as x₀ (source) in the flow matching framework:

**Standard Flow Matching**:
```
x₀ ~ N(0, I)  →  x₁ (data)
```

**FFJORD-based Flow Matching**:
```
x₀ ~ FFJORD(data)  →  x₁ (data)
```

Benefits:
- Source distribution is closer to target (shorter flow path)
- Potentially faster convergence during training

Trade-offs:
- FFJORD sampling is slower than Gaussian
- Requires pre-training FFJORD model
- Additional model to store and deploy

## References

- Original FFJORD paper: [arxiv.org/abs/1810.01367](https://arxiv.org/abs/1810.01367)
- Neural ODE paper (Chen et al. 2018): [arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
- torchdiffeq library: [github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Notes

- The checkpoint at `../../data/ffjord_ckpt.pth` was trained for 400 epochs and is ready to use
- FFJORD samples can be used directly as source distributions in flow matching
- The multiscale architecture is crucial for good performance on images
- Rademacher estimator is recommended for faster training (vs. exact trace computation)
