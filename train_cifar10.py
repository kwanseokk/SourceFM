# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import sys
import joblib
import random
import glob
import numpy as np
import json

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tqdm import trange
from utils_cifar import ema, infiniteloop, generate_samples
from setup_source import setup_source
import torch.nn.functional as F
from dataset import CIFAR10WithClusters, CIFAR10WithClustersFlip

from transport.ot import OTConditionalPlanSampler

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher
)
from torchcfm.models.unet.unet import UNetModelWrapper

class ScaleToNorm:
    def __init__(self, target_norm=20.0, eps=1e-8):
        self.target_norm = target_norm
        self.eps = eps

    def __call__(self, tensor):
        # tensor: C×H×W, float tensor
        norm = tensor.norm() + self.eps
        return tensor / norm * self.target_norm

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_float("sigma", 0.0, help="sigma scale for flow path")
flags.DEFINE_integer(
    "total_steps", 200001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 512, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_float("threshold", 0.06, help="sampling rejection threshold for pruned sampling")
flags.DEFINE_string("subname", '', help="subname of save directory") # ['None', 'mle', 'gmm']
flags.DEFINE_boolean("parallel", False, help="multi gpu training")

# Source Distribution
flags.DEFINE_string("source", 'gaussian', help="source distribution") # ['gaussian', 'norm_gauss', 'mle', 'gmm', 'dct', 'ffjord', 'vonmises' (requires --vmf_all=True), 'sknn_vmf', 'pca_pruned', 'pca_pruned_norm', 'gmm_kmeans', 'gmm_pca', 'mix_vmf_normchi']

# DCT Setting
flags.DEFINE_boolean("x1_process", True, help="process x1 with dct or not")
flags.DEFINE_string("mask", 'weak', help="mask type") # ['weak', 'strong']


# Norm Align Variants
flags.DEFINE_boolean('X1_ChiSphere', False, help="use chi sphere x1 or not")
flags.DEFINE_boolean('X1_Norm', False, help="normalize x1 or not")
flags.DEFINE_boolean('X1_NormAlign', False, help="Norm Align x1 or not")
flags.DEFINE_boolean('X0_ChiSphere', False, help="use chi sphere x0 or not")
flags.DEFINE_boolean('X0_NormAlign', False, help="Norm Align x0 or not")
flags.DEFINE_boolean('X0_Norm', False, help="normalize x0 or not")

# E.T.C
flags.DEFINE_boolean('flip', True, help="use flip or not")
flags.DEFINE_boolean('normalize', True, help="normalize or not")
flags.DEFINE_integer("ncluster", 3, help="cluster or component number of source distribution")
flags.DEFINE_boolean('vmf_all', False, help="use each x1 sample as mu of vmf")
flags.DEFINE_integer('kappa', 0, help="kappa of vmf")
flags.DEFINE_boolean('cls', False, help="use cluster label or not")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    10000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

# use_cuda = torch.backends.mps.is_available()
# device = torch.device("mps" if use_cuda else "cpu")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    if FLAGS.warmup == 0:
        return 1
    else:
        return min(step, FLAGS.warmup) / FLAGS.warmup
    
def ChiSphere(x, d=3072):
    from torch.distributions.chi2 import Chi2
    chi2 = Chi2(d)
    x = x / x.norm(p=2, dim=(1, 2, 3), keepdim=True)
    norms = torch.sqrt(chi2.sample((x.shape[0],))).to(device).view(-1, 1, 1, 1)
    x = x * norms
    return x

def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )
    print(
        f"device = {device}"
    )
    if FLAGS.source.startswith('sknn'):
        sknn = torch.load(f'./data/spherical_knn{FLAGS.ncluster}.pt')
    
    if FLAGS.flip:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        # path corrected by Kwanseok
        root="data",
        train=True,
        download=True,
        transform= transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )
    if FLAGS.source.startswith('sknn'):
        if FLAGS.flip:
            dataset = CIFAR10WithClustersFlip(
                root="data",
                train=True,
                cluster_path=f'./data/spherical_kmeans{FLAGS.ncluster}_flip.pt'
            )
        else:
            dataset = CIFAR10WithClusters(
                root="data",
                train=True,
                transform=transform,
                download=True,
                clusters_path=f'./data/clusters_spherical_knn{FLAGS.ncluster}.pt'
            )
        rank = 1
    elif FLAGS.source == 'gmm_kmeans' or FLAGS.source == 'gmm_pca':
        method = FLAGS.source.split("_")[1]
        if method == 'kmeans':
            dataset = CIFAR10WithClustersFlip(cluster_path=f'./data/spherical_kmeans{FLAGS.ncluster}_flip.pt')
        elif method == 'pca':
            dataset = CIFAR10WithClustersFlip(cluster_path=f'./data/pca{FLAGS.ncluster}_flip.pt')
        rank = 1
    else:
        rank = 0
        
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader, rank=rank)

    # MODELS
    net_model = UNetModelWrapper(
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
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))
    
    # setup source
    src_dict = setup_source(FLAGS, device)
    sampler = src_dict['sampler']
    p_mean = src_dict['mean']
    p_sigma = src_dict['std']
    
        
    #################################
    #            OT-CFM
    #################################
    sbcfm = False
    sigma = FLAGS.sigma
    if FLAGS.model == "otcfm":
        # Using exact method as default
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma, method="exact")
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "sbcfm":
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )
    if FLAGS.source == 'gmm' or FLAGS.source.startswith("sknn"):
        subname = "_" + FLAGS.source + str(FLAGS.ncluster) + f"{FLAGS.subname}"
    else:
        subname = "_" + FLAGS.source + f"{FLAGS.subname}"
    savedir = FLAGS.output_dir + FLAGS.model + subname + "/"
    os.makedirs(savedir, exist_ok=True)
    
    # save args
    flags_dict = FLAGS.flag_values_dict()
    with open(os.path.join(savedir, "args.json"), "w") as f:
        json.dump(flags_dict, f, indent=4)
    
    ####### Resume code #######
    pattern = os.path.join(savedir, f"{FLAGS.model}_cifar10_weights_step_*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print("No checkpoints found")
        start_step = 0
    else:
        checkpoint_steps = [
        (ckpt, int(os.path.basename(ckpt).split("_")[-1].split(".")[0])) for ckpt in checkpoints
        ]
        latest_checkpoint = max(checkpoint_steps, key=lambda x: x[1])[0]
        print(f"Resuming from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        start_step = checkpoint["step"]
        
        # FLAGS.total_steps -= start_step
    ############################
    
    # with trange(FLAGS.total_steps, initial=start_step, dynamic_ncols=True) as pbar:
    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            if rank == 1:
                x1, cluster = next(datalooper)
                x1 = x1.to(device)
                clusters = cluster.to(device)
            else:
                x1 = next(datalooper).to(device)
            
            # X1 processing 
            if FLAGS.X1_Norm:
                x1 = x1 / x1.norm(p=2, dim=(1, 2, 3), keepdim=True)
                # x1 = x1 * 55.4185
                x1 = x1 * FLAGS.scale
            if FLAGS.X1_NormAlign:
                x1 = x1 * (55.4185 / 27.1981)    
            if FLAGS.X1_ChiSphere:
                x1 = ChiSphere(x1)
            if FLAGS.source == 'dct' and FLAGS.x1_process:
                x1 = x1.permute(0, 2, 3, 1)
                x1 = x1 / 2 + 0.5
                x1 = sampler.process_image_torch(x1, 1)
                x1 = x1.permute(0, 3, 1, 2)

            # X0 sampling
            if FLAGS.source in ['gmm', 'dct', 'ffjord']:
                x0 = sampler.sample(x1.shape[0])
            elif FLAGS.source == 'vonmises':
                if not FLAGS.vmf_all:
                    raise ValueError("vonmises source requires --vmf_all=True flag. The non-vmf_all version is deprecated.")
                x0 = sampler.sample_vmf(x1)
                if FLAGS.X0_ChiSphere:
                    x0 = ChiSphere(x0)
            elif FLAGS.source == 'pruned':
                x0 = sampler.sample(x1)
            elif FLAGS.source.startswith('sknn'):
                x0 = sampler.sample(cluster_number=clusters)
                if FLAGS.X0_ChiSphere:
                    x0 = ChiSphere(x0)
            elif FLAGS.source == "pca_pruned":
                x0 = sampler.sample(x1.shape[0], norm=False)
            elif FLAGS.source == "pca_pruned_norm":
                x0 = sampler.sample(x1.shape[0], norm=True)
            elif FLAGS.source == "pca_acceptance":
                x0 = sampler.sample(x1.shape[0], norm=False, reverse=True)
            elif FLAGS.source == "norm_gauss":
                x0 = torch.randn_like(x1)
                x0 = ChiSphere(x0)
            elif FLAGS.source.startswith('gmm_'):
                x0 = sampler.sample_from_cluster(x1.shape[0], cluster_idx=clusters)
                x0 = ChiSphere(x0)
            elif FLAGS.source == "mix_vmf_normchi":
                x0 = sampler.sample_from_cluster(x1.shape[0], cluster_idx=clusters)
                x0 = ChiSphere(x0)
            elif FLAGS.source == "mix_vmf_normchi":
                mask = torch.rand(x1.shape[0], device=device) < 0.5
                x0 = torch.zeros_like(x1)
                if mask.any():
                    x0_method1 = torch.randn_like(x1[mask])
                    x0_method1 = x0_method1 / x0_method1.norm(p=2, dim=(1,2,3), keepdim=True)
                    # x0_method1 = x0_method1 * norms[mask]
                    x0[mask] = x0_method1
                if (~mask).any():
                    x0_method2 = sampler.sample_vmf_mukappa(x1[~mask], torch.tensor([FLAGS.kappa]*x1[~mask].shape[0]).to(device))
                    x0[~mask] = x0_method2
                x0 = ChiSphere(x0)
            else:
                x0 = torch.randn_like(x1)
            if FLAGS.source == 'mle':
                x0 = p_mean + p_sigma * x0
            if FLAGS.X0_NormAlign:
                x0 = x0 * 27.1981 / 55.4185
            if FLAGS.X0_Norm:
                x0 = x0 / x0.norm(p=2, dim=(1, 2, 3), keepdim=True)
                x0 = x0 * 55.4185
            
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            
            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, savedir, step, sampler=sampler, net_="normal", FLAGS=FLAGS, device=device)
                generate_samples(ema_model, savedir, step, sampler=sampler, net_="ema", FLAGS=FLAGS, device=device)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
