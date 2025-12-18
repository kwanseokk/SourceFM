import os
import sys
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from accelerate import Accelerator
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.models.unet.unet import UNetModelWrapper
from PIL import Image
from tqdm import tqdm
import joblib, math
from torchvision.utils import save_image
from utils_cifar import gen_img
from norm.norm_pred import NormPredictorCNN
import numpy as np
from setup_source import setup_source
import copy

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device
rank = accelerator.local_process_index
dtype = torch.float32

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results/otcfm_None", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")

# Source Distribution
flags.DEFINE_string("source", "", help="source distribution type. choose in [gaussian, gmm, dct]")

# DCT Setting
flags.DEFINE_boolean("x1_process", False, help="process x1 with dct or not")
flags.DEFINE_string("mask", 'weak', help="mask type") # ['weak', 'strong']

# ODE
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "euler", help="integration method to use")
flags.DEFINE_integer("step", 10000, help="training steps")
flags.DEFINE_integer("start_step", 0, help="training steps")
flags.DEFINE_integer("end_step", 200000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 512, help="Batch size to compute FID")

# Pruning
flags.DEFINE_float("threshold", 0.06, help="sampling pruned threshold for pruned sampling")

# Norm Align Variants
flags.DEFINE_boolean('X1_ChiSphere', False, help="use chi sphere x1 or not")
flags.DEFINE_boolean('X1_Norm', False, help="normalize x1 or not")
flags.DEFINE_boolean('X1_NormAlign', False, help="Norm Align x1 or not")
flags.DEFINE_boolean('X0_ChiSphere', False, help="use chi sphere x0 or not")
flags.DEFINE_boolean('X0_NormAlign', False, help="Norm Align x0 or not")
flags.DEFINE_boolean('X0_Norm', False, help="normalize x0 or not")

# E.T.C
flags.DEFINE_integer("ncluster", 2, help="cluster or component number of source distribution")
flags.DEFINE_boolean("vmf_all", False, help="use all clusters for vmf sampling")
flags.DEFINE_integer("kappa", 0, help="kappa for vmf sampling")
flags.DEFINE_boolean("cls", False, help="use cls for vmf sampling")
flags.DEFINE_boolean("normalize", True, help="use vmf sampling")
flags.DEFINE_boolean("flip", False, help="use flip for vmf sampling")

FLAGS(sys.argv)
print("Integration steps:", FLAGS.integration_steps)

if FLAGS.vmf_all or FLAGS.source == 'pruned':
    from torchvision import datasets, transforms
    trainset = datasets.CIFAR10(
            # path corrected by Kwanseok
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
    data, _ = next(iter(trainloader))
    
# if rank ==0:
    # print(FLAGS)
if accelerator.process_index == 0:
    print(f"Source Distribution {FLAGS.source.upper()}")
    
def ChiSphere(x, d=3072):
    from torch.distributions.chi2 import Chi2
    chi2 = Chi2(d)
    x = x / x.norm(p=2, dim=(1, 2, 3), keepdim=True)
    norms = torch.sqrt(chi2.sample((x.shape[0],))).to(device).view(-1, 1, 1, 1)
    x = x * norms
    return x
    
def evaluate_model(checkpoint_path, sample_folder_dir, model_type="ema_model"):
    src_dict = setup_source(FLAGS, device)
    sampler = src_dict["sampler"]
    p_mean = src_dict["mean"]
    p_sigma = src_dict["std"]
    
    total_samples, n = FLAGS.num_gen, FLAGS.batch_size_fid
    global_batch_size = n * accelerator.num_processes
    total_samples = int(math.ceil(total_samples / global_batch_size) * global_batch_size)
    # number of gpus
    num_gpus = accelerator.num_processes
    if num_gpus == 1:
        print("Single GPU mode")
        total_samples = 51200
        print(f"Generating {total_samples} samples")
    elif accelerator.process_index == 0:
        print(f"Generating {total_samples} samples")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    new_net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_classes=FLAGS.ncluster if FLAGS.cls else None,
        class_cond=FLAGS.cls,
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
        
    for i in pbar:
        samples = gen_img(FLAGS, sampler, FLAGS.batch_size_fid, new_net, node, device)
        # total = accelerator.num_processes * FLAGS.batch_size_fid
        for i, sample in enumerate(samples):
            index = i * accelerator.num_processes + accelerator.process_index + total
            sample_np = sample.permute(1, 2, 0).cpu().numpy()
            Image.fromarray(sample_np).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        accelerator.wait_for_everyone()
    return

# Main function to compute FID for all checkpoints
def main(_):
    steps = range(FLAGS.start_step, FLAGS.end_step+1, FLAGS.step)  # From 8000 to 120000, every 8000 steps
    ema_fid_scores = []
    # net_fid_scores = []
    model_type = "ema_model"
    for step in steps:
        save_dir = os.path.join(FLAGS.input_dir, "gen_imgs", f"{model_type}_{step}_{FLAGS.source}")
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            if accelerator.process_index == 0:
                print(f"Saving .png samples at {save_dir}")
        accelerator.wait_for_everyone()
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(FLAGS.input_dir, f"{FLAGS.model}_cifar10_weights_step_{step}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        # Compute FID for ema_model
        ema_score = evaluate_model(checkpoint_path, sample_folder_dir=save_dir, model_type=model_type)
        if rank == 0:
            print(f"Computing FID for {checkpoint_path} ({model_type})")
            score = fid.compute_fid(
                fdir1=save_dir,
                dataset_name="cifar10",
                batch_size=FLAGS.batch_size_fid,
                dataset_res=32,
                num_gen=FLAGS.num_gen,
                dataset_split="train",
                mode="legacy_tensorflow",
            )
            print(f"FID for {checkpoint_path} ({model_type}): {score}")
            ema_fid_scores.append(score)
        accelerator.wait_for_everyone()

        # # Compute FID for net_model
    
    print("EMA FID Scores:", ema_fid_scores)
    # print("Net FID Scores:", net_fid_scores)

if __name__ == "__main__":
    app.run(main)