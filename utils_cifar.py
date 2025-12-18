from torchvision.utils import make_grid, save_image
from torchdyn.core import NeuralODE
import copy
import torch
import numpy as np
from torchdiffeq import odeint


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

def ChiSphere(x, d=3072):
    from torch.distributions.chi2 import Chi2
    chi2 = Chi2(d)
    x = x / x.norm(p=2, dim=(1, 2, 3), keepdim=True)
    norms = torch.sqrt(chi2.sample((x.shape[0],))).to(device).view(-1, 1, 1, 1)
    x = x * norms
    return x

def ema(source, target, decay):
    
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def gen_img(FLAGS, sampler, num, new_net, node, device, unused_latent=None, visualize=False):
    if visualize:
        integration_steps = 100
        integration_method = "euler"
    else:
        integration_steps = FLAGS.integration_steps
        integration_method = FLAGS.integration_method
    
    with torch.no_grad():
        if FLAGS.source == "gaussian":
            x = torch.randn(num, 3, 32, 32, device=device)
            if FLAGS.X0_NormAlign:
                x = x * (27.1981 / 55.4185)
            if FLAGS.X0_Norm:
                x = x / x.norm(p=2, dim=(1, 2, 3), keepdim=True)
                x = x * 55.4185
        elif FLAGS.source in ["gmm", "dct", "ffjord"]:
            x = sampler.sample(num)
        elif FLAGS.source == "vonmises":
            if not FLAGS.vmf_all:
                raise ValueError("vonmises source requires --vmf_all=True flag. The non-vmf_all version is deprecated.")
            images = data[torch.randint(0, 50000, (num,))].to(device)
            if FLAGS.flip:
                # randomly horizontal flip for each image
                do_flip = torch.rand(images.size(0), device=images.device) < 0.5
                images[do_flip] = torch.flip(images[do_flip], [3])
            x = sampler.sample_vmf(images)
            x = ChiSphere(x)
        elif FLAGS.source == 'pruned':
            x1 = data[torch.randint(0, 50000, (num,))].to(device)
            if FLAGS.flip:
                do_flip = torch.rand(x1.size(0), device=x1.device) < 0.5
                x1[do_flip] = torch.flip(x1[do_flip], [3])
            x = sampler.sample(x1)
        elif FLAGS.source.startswith('sknn'):
            x = sampler.sample(num)
            if FLAGS.X1_ChiSphere:
                x = ChiSphere(x)
        elif FLAGS.source == "pca_pruned":
            x = sampler.sample(num, norm=False)
        elif FLAGS.source == "pca_pruned_norm":
            x = sampler.sample(num, norm=True)
        elif FLAGS.source == "pca_acceptance":
            x = sampler.sample(num, norm=False, reverse=True)
        elif FLAGS.source == "norm_gauss":
            x = torch.randn(num, 3, 32, 32, device=device)
            x = ChiSphere(x)
        elif FLAGS.source.startswith('gmm_'):
            x, cluster = sampler.sample(num, return_cluster=True)
            x = ChiSphere(x)
        # else:
            # raise(f"No source distribution type {FLAGS.source}")
        
        if FLAGS.cls and integration_method == "euler":
            def ode_func(t, x, args=None):
                return new_net(t, x, y=clusters, **(args or {}))    
            node_ = NeuralODE(ode_func, solver="euler", sensitivity="adjoint")
            t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
            traj = node_.trajectory(x, t_span=t_span)
        elif integration_method == "euler":
            t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
    traj = traj[-1, :]
    
    if FLAGS.X1_Norm or FLAGS.X1_ChiSphere:
        traj = traj / traj.norm(p=2, dim=(1, 2, 3), keepdim=True) * 27.1981
        # traj *= (27.1981 / FLAGS.scale)
    if FLAGS.X1_NormAlign:
        traj = traj * (27.1981 / 55.4185)
        
    if visualize:
        traj = traj.view([-1, 3, 32, 32]).clip(-1, 1)
        img = traj / 2 + 0.5
    else:
        img = torch.clamp(127.5 * traj + 128.0, 0, 255).to(torch.uint8)
    return img

def infiniteloop(dataloader, rank=0):
    while True:
        for x, y in iter(dataloader):
            if rank == 0:
                yield x
            else:
                yield x, y
                
                
def generate_samples(model, save_dir, step, sampler, net_, FLAGS, device):
    model.eval()
    
    model_ = copy.deepcopy(model)
    if FLAGS.parallel:
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    img = gen_img(FLAGS, sampler, 64, model_, node_, device, visualize=True)

    save_image(img, save_dir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()