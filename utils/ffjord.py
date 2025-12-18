import argparse
import torch

import utils.lib.layers as layers
import utils.lib.utils as utils
import utils.lib.odenvp as odenvp
import utils.lib.multiscale_parallel as multiscale_parallel

from utils.train_misc import create_regularization_fns, get_regularization

# Use PyTorch 2.0's compiler (optional)
TORCH_COMPILE = True  # Set to True if using PyTorch 2.0+

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
# Configuration dictionary instead of argparse
CONFIG = {
    "data": "cifar10",  # Dataset choice: "mnist", "svhn", "cifar10", "lsun_church"
    "dims": "64,64,64",  # Hidden layer dimensions
    "strides": "1,1,1,1",  # Stride settings for convolutional layers
    "num_blocks": 2,  # Number of stacked CNFs
    "conv": True,  # Use convolutional layers
    "layer_type": "concat",  # Type of layer transformation
    "divergence_fn": "approximate",  # Method for divergence computation
    "nonlinearity": "softplus",  # Activation function
    "solver": "dopri5",  # ODE solver type
    "atol": 1e-5,  # Absolute tolerance for solver
    "rtol": 1e-5,  # Relative tolerance for solver
    "step_size": None,  # Optional fixed step size for solver
    "imagesize": None,  # Image size (if needed)
    "alpha": 1e-6,  # Logit transform parameter
    "time_length": 1.0,  # Maximum time for ODE integration
    "train_T": True,  # Whether to train time parameter T
    "num_epochs": 1000,  # Number of training epochs
    "batch_size": 512,  # Training batch size
    "test_batch_size": 200,  # Testing batch size
    "lr": 1e-3,  # Learning rate
    "warmup_iters": 1000,  # Warmup iterations for training
    "weight_decay": 0.0,  # Weight decay for regularization
    "spectral_norm_niter": 10,  # Iterations for spectral normalization
    "add_noise": True,  # Add noise to training data
    "batch_norm": False,  # Use batch normalization
    "residual": False,  # Use residual connections
    "autoencode": False,  # Use autoencoder structure
    "rademacher": True,  # Use Rademacher distribution for noise
    "spectral_norm": False,  # Apply spectral normalization
    "multiscale": True,  # Use multiscale architecture
    "parallel": False,  # Use parallel CNF layers
    "max_grad_norm": 1e10,  # Gradient clipping threshold
    "save": "experiments/cnf",  # Directory to save models
}

class FFJORD:
    def __init__(self, device='cuda', ckpt_path=None):
        self.device = device
        
        self.data_shape = (3, 32, 32)
        
        # Enable cuDNN optimization
        torch.backends.cudnn.benchmark = True
        regularization_fns, regularization_coeffs = create_regularization_fns(CONFIG)
        self.model = self.create_model(CONFIG, self.data_shape, regularization_fns)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        
        if ckpt_path:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['state_dict'])
            print(f"Model loaded from {ckpt_path}")
            
        # Optional: Compile model for faster execution (PyTorch 2.0+)
        # if TORCH_COMPILE:
        #     self.model = torch.compile(self.model)

    # args input removed
    def create_model(self, config, data_shape, regularization_fns):
        hidden_dims = tuple(map(int, config["dims"].split(",")))
        strides = tuple(map(int, config["strides"].split(",")))

        if config["multiscale"]:
            model = odenvp.ODENVP(
                (config["batch_size"], *data_shape),
                n_blocks=config["num_blocks"],
                intermediate_dims=hidden_dims,
                nonlinearity=config["nonlinearity"],
                alpha=config["alpha"],
                cnf_kwargs={"T": config["time_length"], "train_T": config["train_T"], "regularization_fns": regularization_fns},
            )
        elif config["parallel"]:
            model = multiscale_parallel.MultiscaleParallelCNF(
                (config["batch_size"], *data_shape),
                n_blocks=config["num_blocks"],
                intermediate_dims=hidden_dims,
                alpha=config["alpha"],
                time_length=config["time_length"],
            )
        else:
            if config[autoencode]:

                def build_cnf():
                    autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                        hidden_dims=hidden_dims,
                        input_shape=data_shape,
                        strides=strides,
                        conv=config["conv"],
                        layer_type=config["layer_type"],
                        nonlinearity=config["nonlinearity"],
                    )
                    odefunc = layers.AutoencoderODEfunc(
                        autoencoder_diffeq=autoencoder_diffeq,
                        divergence_fn=config["divergence_fn"],
                        residual=config["residual"],
                        rademacher= config["rademacher"],
                    )
                    cnf = layers.CNF(
                        odefunc=odefunc,
                        T=config["time_length"],
                        regularization_fns=regularization_fns,
                        solver=config["solver"],
                    )
                    return cnf
            else:

                def build_cnf():
                    diffeq = layers.ODEnet(
                        hidden_dims=hidden_dims,
                        input_shape=data_shape,
                        strides=strides,
                        conv=config["conv"],
                        layer_type=config["layer_type"],
                        nonlinearity=config["nonlinearity"],
                    )
                    odefunc = layers.ODEfunc(
                        diffeq=diffeq,
                        divergence_fn=config["divergence_fn"],
                        residual=config["residual"],
                        rademacher=config["rademacher"],
                    )
                    cnf = layers.CNF(
                        odefunc=odefunc,
                        T=config["time_length"],
                        train_T=config["train_T"],
                        regularization_fns=regularization_fns,
                        solver=config["solver"],
                    )
                    return cnf

            chain = [layers.LogitTransform(alpha=config["alpha"])] if config["alpha"] > 0 else [layers.ZeroMeanTransform()]
            chain = chain + [build_cnf() for _ in range(config["num_blocks"])]
            if config["batch_norm"]:
                chain.append(layers.MovingBatchNorm2d(data_shape[0]))
            model = layers.SequentialFlow(chain)
        return model

    # def sample(self, num_samples):
        
    #     cvt = lambda x: x.type(torch.float32).to(self.device, non_blocking=True)
    #     x = cvt(torch.randn(num_samples, *self.data_shape))
    #     x = self.model(x, reverse=True)
        
    #     return x
    @torch.no_grad()  # Disable autograd for faster inference
    def sample(self, num_samples):
        # Generate noise directly on GPU and use autocast for mixed precision
        # with torch.autocast(device_type=self.device.type, dtype=torch.float32):
        x = torch.randn(num_samples, *self.data_shape, device=self.device)
        x = self.model(x, reverse=True)
        x = (x - 0.5) * 2.0  # Scale to [-1, 1] range
        return x

if __name__ == '__main__':
    model = model_nf()
    model.to('cuda')
    model.eval()
    x0 = torch.randn(512, 3, 32, 32).to('cuda')
    
    x0 = model(x0, reverse=True)
    print(x0.shape)
    