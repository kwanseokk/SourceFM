import os
import sys
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.models.unet.unet import UNetModelWrapper
import joblib
import random
import numpy as np
import cv2
from joblib import Parallel, delayed
import torch.nn.functional as F

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results/otcfm_dctmask_strong6n_torch", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")
flags.DEFINE_string("source", "", help="source of the latent space")

FLAGS(sys.argv)

# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def create_dct_matrix(N):
    """
    Create an NxN DCT transformation matrix using vectorized operations.
    """
    n = torch.arange(N, dtype=torch.float32)  # Ensure float32 for precision
    k = n.view(-1, 1)  # Reshape for broadcasting
    coeffs = torch.sqrt(torch.tensor(2 / N)) * torch.ones(N, dtype=torch.float32)
    coeffs[0] = torch.sqrt(torch.tensor(1 / N))  # First row normalization

    dct_mat = coeffs[:, None] * torch.cos(torch.pi * k * (n + 0.5) / N)
    return dct_mat

DCT_MAT = create_dct_matrix(8)
DCT_MAT = DCT_MAT.to(device)
DCT_T = DCT_MAT.T

# # Load the masks once
lum_mask = torch.tensor(np.load('../../data/luminance_mask.npy'), dtype=torch.bool, device=device)
chro_cr_mask = torch.tensor(np.load('../../data/chrominance_cr_mask.npy'), dtype=torch.bool, device=device)
chro_cb_mask = torch.tensor(np.load('../../data/chrominance_cb_mask.npy'), dtype=torch.bool, device=device)

dct_stats = torch.load("pretrain_weight/noise_dctmask6_strong_torch.pt")
noise_mean, noise_std = dct_stats['mean'].unsqueeze(0).mean([2, 3], keepdim=True).to(device), dct_stats['std'].unsqueeze(0).mean([2, 3], keepdim=True).to(device)

# dct_stats = torch.load("pretrain_weight/noise_dctmask6_strong.pt")
# mean, std = dct_stats['mean'].unsqueeze(0).mean([2, 3], keepdim=True).to(device), dct_stats['std'].unsqueeze(0).mean([2, 3], keepdim=True).to(device)
def rgb_to_ycrcb(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts an RGB image (uint8 range [0, 255]) to YCrCb format.
    :param rgb: Input tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.uint8
    :return: Converted tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.float32
    """
    # Ensure input is uint8
    # assert rgb.dtype == torch.uint8, "Input must be in uint8 format (0-255)."
    
    # Convert to float32 for precision
    # rgb = rgb.to(torch.float32)
    
    # RGB channels
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Compute Y, Cr, and Cb
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    # Stack to get final tensor
    ycrcb = torch.stack([Y, Cr, Cb], dim=-1)

    return ycrcb

def ycrcb_to_rgb(ycrcb: torch.Tensor) -> torch.Tensor:
    """
    Converts a YCrCb image (float32) back to RGB format.
    :param ycrcb: Input tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.float32
    :return: Converted tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.uint8
    """
    # Ensure input is float32
    assert ycrcb.dtype == torch.float32, "Input must be in float32 format."

    # Extract Y, Cr, Cb channels
    Y, Cr, Cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]

    # Compute R, G, B
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)

    # Stack and clip to [0, 255] range
    rgb = torch.stack([R, G, B], dim=-1)

    return rgb

def process_image_torch(x0_img):
    # Scale to 0-255 and convert to uint8 tensor
    B, _, _, _ = x0_img.shape
    x0_scaled = (x0_img).clamp(0, 1)

    ycrcb = rgb_to_ycrcb(x0_scaled)
    # print(ycrcb)

    # Extract channels
    lum, cr, cb = ycrcb.unbind(dim=-1)

    # Reshape into 8×8 patches
    patches_lum = lum.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(B, -1, 8, 8)
    patches_cr = cr.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(B, -1, 8, 8)
    patches_cb = cb.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(B, -1, 8, 8)
    # patches_lum = lum.unfold(1, 8, 8).unfold(2, 8, 8)
    # patches_cr = cr.unfold(1, 8, 8).unfold(2, 8, 8)
    # patches_cb = cb.unfold(1, 8, 8).unfold(2, 8, 8)
    
    # # Apply DCT
    dct_lum = torch.matmul(DCT_MAT, torch.matmul(patches_lum, DCT_T))
    dct_cr = torch.matmul(DCT_MAT, torch.matmul(patches_cr, DCT_T))
    dct_cb = torch.matmul(DCT_MAT, torch.matmul(patches_cb, DCT_T))
    # dct_lum = torch.einsum('ij,bxjk,kl->bxil', DCT_MAT, patches_lum, DCT_T)
    # dct_cr = torch.einsum('ij,bxjk,kl->bxil', DCT_MAT, patches_cr, DCT_T)
    # dct_cb = torch.einsum('ij,bxjk,kl->bxil', DCT_MAT, patches_cb, DCT_T)
    

    # Apply masks (ensure they are on the same device)
    dct_lum *= ~lum_mask
    dct_cr *= ~chro_cr_mask
    dct_cb *= ~chro_cb_mask

    # Inverse DCT
    recon_lum = torch.matmul(DCT_T, torch.matmul(dct_lum, DCT_MAT))
    recon_cr = torch.matmul(DCT_T, torch.matmul(dct_cr, DCT_MAT))
    recon_cb = torch.matmul(DCT_T, torch.matmul(dct_cb, DCT_MAT))
    # recon_lum = torch.einsum('ij,bxjk,kl->bxil', DCT_T, dct_lum, DCT_MAT)
    # recon_cr = torch.einsum('ij,bxjk,kl->bxil', DCT_T, dct_cr, DCT_MAT)
    # recon_cb = torch.einsum('ij,bxjk,kl->bxil', DCT_T, dct_cb, DCT_MAT)

    # Reconstruct from patches
    # reconstructed_lum = recon_lum.permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
    # reconstructed_cr = recon_cr.permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
    # reconstructed_cb = recon_cb.permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
    # Reconstruct from patches
    reconstructed_lum = recon_lum.view(B, 4, 4, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
    reconstructed_cr = recon_cr.view(B, 4, 4, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
    reconstructed_cb = recon_cb.view(B, 4, 4, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 32, 32)

    # Stack channels back together
    reconstructed_ycrcb = torch.stack([reconstructed_lum, reconstructed_cr, reconstructed_cb], dim=-1)

    # Convert back to RGB
    reconstructed_rgb = ycrcb_to_rgb(reconstructed_ycrcb)

    # Normalize to -1 to 1 range
    return ((reconstructed_rgb - 0.5) * 6).to(device)


# Function to load and evaluate a model
def evaluate_model(checkpoint_path, model_type="ema_model"):
    new_net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint[model_type]
    try:
        new_net.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        new_net.load_state_dict(new_state_dict)
    new_net.eval()

    # Define the integration method if euler is used
    if FLAGS.integration_method == "euler":
        node = NeuralODE(new_net, solver=FLAGS.integration_method)

    def gen_1_img(unused_latent):
        with torch.no_grad():
            # x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
            if FLAGS.source == "dct":
                x0 = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
                x0 = x0 / 6 + 0.5
                x0 = x0.permute(0, 2, 3, 1)
                x0 = process_image_torch(x0)
                x = x0.permute(0, 3, 1, 2)
                x = (x - noise_mean) / noise_std
            else:
                x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)

            if FLAGS.integration_method == "euler":
                t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
                traj = node.trajectory(x, t_span=t_span)
            else:
                t_span = torch.linspace(0, 1, 2, device=device)
                traj = odeint(
                    new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
                )
        
        traj = traj[-1, :]
        # traj = (traj * 0.5 + 0.5)
        # img = (traj * 255).clip(0, 255).to(torch.uint8)
        img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
        return img

    print(f"Computing FID for {checkpoint_path} ({model_type})")
    score = fid.compute_fid(
        gen=gen_1_img,
        dataset_name="cifar10",
        batch_size=FLAGS.batch_size_fid,
        dataset_res=32,
        num_gen=FLAGS.num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )
    print(f"FID for {checkpoint_path} ({model_type}): {score}")
    return score

# Main function to compute FID for all checkpoints
def main(_):
    steps = range(70000, 200001, 10000)  # From 8000 to 120000, every 8000 steps
    ema_fid_scores = []
    net_fid_scores = []

    for step in steps:
        checkpoint_path = os.path.join(FLAGS.input_dir, f"otcfm_cifar10_weights_step_{step}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        # Compute FID for ema_model
        ema_score = evaluate_model(checkpoint_path, model_type="ema_model")
        ema_fid_scores.append(ema_score)

        # Compute FID for net_model
        net_score = evaluate_model(checkpoint_path, model_type="net_model")
        net_fid_scores.append(net_score)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(steps[:len(ema_fid_scores)], ema_fid_scores, label="EMA Model", marker="o")
    plt.plot(steps[:len(net_fid_scores)], net_fid_scores, label="Net Model", marker="o")
    # Annotate each point with its actual value
    for step, fid in zip(steps[:len(ema_fid_scores)], ema_fid_scores):
        plt.text(step, fid, f"{fid:.2f}", ha='right', va='bottom', fontsize=10, color="blue")

    for step, fid in zip(steps[:len(net_fid_scores)], net_fid_scores):
        plt.text(step, fid, f"{fid:.2f}", ha='left', va='top', fontsize=10, color="red")
    plt.xlabel("Training Steps")
    plt.ylabel("FID Score")
    plt.title("FID Scores Over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"fid_scores_{FLAGS.source}_dctmask_torch_strong6n.png")
    plt.show()
    
    print("EMA FID Scores:", ema_fid_scores)
    print("Net FID Scores:", net_fid_scores)

if __name__ == "__main__":
    app.run(main)