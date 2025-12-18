import numpy as np
import mpmath
import torch
import torch.nn as nn

import utils.utils as utils

def norm(tensor, dim, keepdim=True):
    return torch.linalg.norm(tensor, dim=dim, keepdim=keepdim)

class vMF(nn.Module):
    
    '''
    vMF(x; mu, kappa)
    '''
    
    def __init__(self, x_dim, reg=1e-6):
        
        super(vMF, self).__init__()
        
        self.x_dim = x_dim
        
        self.mu_unnorm = nn.Parameter(torch.randn(x_dim))
        self.logkappa = nn.Parameter(0.01*torch.randn([]))
        
        self.reg = reg
        
    def set_params(self, mu, kappa):
        
        with torch.no_grad():
            self.mu_unnorm.copy_(mu)
            self.logkappa.copy_(torch.log(kappa+utils.realmin))
    
    def get_params(self):
        
        mu = self.mu_unnorm / utils.norm(self.mu_unnorm)
        kappa = self.logkappa.exp() + self.reg
        
        return mu, kappa
        
    def forward(self, x, utc=False):
        
        '''
        Evaluate logliks, log p(x)
        
        Args:
            x = batch for x
            utc = whether to evaluate only up to constant or exactly 
                if True, no log-partition computed
                if False, exact loglik computed

        Returns:
            logliks = log p(x)
        '''

        mu, kappa = self.get_params()

        dotp = (mu.unsqueeze(0) * x).sum(1)
        
        if utc:
            logliks = kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, kappa)
            logliks = kappa * dotp + logC
        
        return logliks
    
    def sample(self, N=1, rsf=10):
    
        '''
        Args:
            N = number of samples to generate
            rsf = multiplicative factor for extra backup samples in rejection sampling 
        
        Returns:
            samples; N samples generated
            
        Notes:
            no autodiff
        '''
        
        d = self.x_dim
        
        with torch.no_grad():
            
            mu, kappa = self.get_params()
        
            # Step-1: Sample uniform unit vectors in R^{d-1} 
            v = torch.randn(N, d-1).to(mu)
            v = v / utils.norm(v, dim=1)
            
            # Step-2: Sample v0
            kmr = np.sqrt( 4*kappa.item()**2 + (d-1)**2 )
            bb = (kmr - 2*kappa) / (d-1)
            aa = (kmr + 2*kappa + d - 1) / 4
            dd = (4*aa*bb)/(1+bb) - (d-1)*np.log(d-1)
            beta = torch.distributions.Beta( torch.tensor(0.5*(d-1)), torch.tensor(0.5*(d-1)) )
            uniform = torch.distributions.Uniform(0.0, 1.0)
            v0 = torch.tensor([]).to(mu)
            while len(v0) < N:
                eps = beta.sample([1, rsf*(N-len(v0))]).squeeze().to(mu)
                uns = uniform.sample([1, rsf*(N-len(v0))]).squeeze().to(mu)
                w0 = (1 - (1+bb)*eps) / (1 - (1-bb)*eps)
                t0 = (2*aa*bb) / (1 - (1-bb)*eps)
                det = (d-1)*t0.log() - t0 + dd - uns.log()
                # v0 = torch.cat([v0, torch.tensor(w0[det>=0]).to(mu)])
                v0 = torch.cat([v0, w0[det>=0].clone()])
                if len(v0) > N:
                    v0 = v0[:N]
                    break
            v0 = v0.reshape([N,1])
            
            # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
            samples = torch.cat([v0, (1-v0**2).sqrt()*v], 1)
    
            # Setup-4: Householder transformation
            e1mu = torch.zeros(d,1).to(mu);  e1mu[0,0] = 1.0
            e1mu = e1mu - mu if len(mu.shape)==2 else e1mu - mu.unsqueeze(1)
            e1mu = e1mu / utils.norm(e1mu, dim=0)
            samples = samples - 2 * (samples @ e1mu) @ e1mu.t()
    
        return samples
    
    @staticmethod
    def sample_batch(mu_batch, kappa):
        """
        Samples B vectors from B vMF distributions efficiently using vectorization.

        Args:
            mu_batch (torch.Tensor): Batch of mean directions, shape (B, d).
                                     Assumed to be normalized (||mu||=1).
            kappa (torch.Tensor): Concentration parameter (scalar tensor).

        Returns:
            torch.Tensor: Sampled vectors, shape (B, d).
        """
        B, d = mu_batch.shape
        device = mu_batch.device
        kappa_val = kappa.item() # Get scalar value

        # 1. Calculate parameters for Wood/Ulrich rejection sampling based on kappa and d
        # These are computed once as kappa and d are fixed for the batch.
        # Using the formulation from: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf (Appendix)
        # which seems robust.
        # Note: d must be >= 2
        if d < 2:
            raise ValueError("Dimension d must be >= 2 for vMF sampling")

        # Calculate b parameter for rejection sampling
        # Using precise formula involving sqrt
        kmr = np.sqrt(4 * kappa_val**2 + (d - 1)**2)
        b_val = (kmr - 2 * kappa_val) / (d - 1) # Note: Different 'b' than some sources, check derivation if issues arise
        # Alternative b (closer to Wood '94): b = (d - 1) / (2 * kappa + sqrt(4 * kappa^2 + (d - 1)^2))

        # Calculate x0 (or a in some notations) and c for rejection sampling check
        a_val = (kmr + 2 * kappa_val + d - 1) / 4 # Intermediate value 'a'
        # d_val = (4*a_val*b_val)/(1+b_val) - (d-1)*np.log(d-1) # Intermediate value 'd' in original code, seems related to acceptance const 'c'

        # Simplified constants based directly on b_val for the acceptance check form used below
        # (Based on acceptance check: t > (1 - (1+b)*eps) / (1 - (1-b)*eps) )
        # Let's stick to the Beta distribution approach for w:
        beta_dist = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))

        # Constants for acceptance condition (derived from Ulrich '84 / Wood '94 style)
        # Need b = (d - 1) / (2 * kappa + sqrt(4 * kappa**2 + (d - 1)**2)) for this form
        b_wood = (d - 1) / (2 * kappa_val + np.sqrt(4 * kappa_val**2 + (d - 1)**2))
        x0_wood = (1 - b_wood) / (1 + b_wood)
        c_wood = kappa_val * x0_wood + (d - 1) * np.log(1 - x0_wood**2)

        b = torch.tensor(b_wood, device=device)
        x0 = torch.tensor(x0_wood, device=device)
        c = torch.tensor(c_wood, device=device)


        # 2. Vectorized Rejection Sampling for the radial component 'w'
        w_samples = torch.zeros(B, device=device)
        accepted_indices = torch.zeros(B, dtype=torch.bool, device=device)
        num_accepted = 0

        while num_accepted < B:
            num_needed = B - num_accepted
            # Sample candidates only for those still needed
            z_cand = beta_dist.sample((num_needed,)).to(device) # Sample from Beta
            u_cand = torch.rand(num_needed, device=device)      # Sample from Uniform

            # Calculate w candidates using the transformation
            w_cand = (1 - (1 + b) * z_cand) / (1 - (1 - b) * z_cand)

            # Acceptance condition (vectorized)
            accept_crit = kappa * w_cand + (d - 1) * torch.log(1 - x0 * w_cand) - c
            accept_mask = (accept_crit >= torch.log(u_cand))

            # Find which original indices these candidates correspond to
            unaccepted_indices = (~accepted_indices).nonzero(as_tuple=True)[0]
            current_accepted_indices = unaccepted_indices[accept_mask]

            # Store accepted samples
            if len(current_accepted_indices) > 0:
                w_samples[current_accepted_indices] = w_cand[accept_mask]
                accepted_indices[current_accepted_indices] = True
                num_accepted += len(current_accepted_indices)

        w = w_samples.unsqueeze(1) # Shape: [B, 1]

        # 3. Sample the remaining d-1 dimensions uniformly on the unit sphere S^(d-2)
        v_unnorm = torch.randn(B, d - 1, device=device)
        v_norm = norm(v_unnorm, dim=1)
        # Handle potential zero norm vectors
        v_norm = torch.where(v_norm == 0, torch.ones_like(v_norm), v_norm)
        v = v_unnorm / v_norm # Shape: [B, d-1]

        # 4. Combine w and v to form samples aligned with the first basis vector e1
        # Ensure w is within [-1, 1] for sqrt
        w = torch.clamp(w, -1.0 + utils.realmin, 1.0 - utils.realmin)
        samples_e1 = torch.cat([w, torch.sqrt(1 - w**2) * v], dim=1) # Shape: [B, d]

        # 5. Rotate samples from e1 alignment to mu_batch alignment using Householder reflection
        e1 = torch.zeros(1, d, device=device)
        e1[0, 0] = 1.0 # Shape: [1, d]

        u = e1 - mu_batch # Reflection vector, shape: [B, d]
        u_norm = norm(u, dim=1)
        # Avoid division by zero if mu_batch is already e1
        u_norm = torch.where(u_norm < utils.realmin, torch.ones_like(u_norm), u_norm)
        householder_v = u / u_norm # Normalized reflection vector, shape: [B, d]

        # Apply reflection: x' = x - 2 * dot(x, v) * v
        dot_prod = (samples_e1 * householder_v).sum(dim=1, keepdim=True) # Shape: [B, 1]
        samples = samples_e1 - 2 * dot_prod * householder_v # Shape: [B, d]

        # Normalize samples to ensure they are exactly on the unit sphere due to potential float errors
        samples = samples / norm(samples, dim=1)

        return samples
    
    # @staticmethod
    # def sample_batch_mukappa(mu_batch, kappa_batch):
    #     """
    #     Samples B vectors efficiently from B vMF distributions, each with its own
    #     mean direction mu and concentration kappa.

    #     Args:
    #         mu_batch (torch.Tensor): Batch of mean directions, shape (B, d).
    #                                 Assumed to be normalized (||mu||=1).
    #         kappa_batch (torch.Tensor): Batch of concentration parameters, shape (B,).
    #                                     All kappas must be > 0.

    #     Returns:
    #         torch.Tensor: Sampled vectors, shape (B, d).
    #     """
    #     B, d = mu_batch.shape
    #     device = mu_batch.device

    #     if not torch.is_tensor(kappa_batch) or kappa_batch.ndim != 1 or kappa_batch.shape[0] != B:
    #         raise ValueError(f"kappa_batch must be a 1D tensor of shape ({B},), but got shape {kappa_batch.shape}")
    #     if torch.any(kappa_batch <= 0):
    #         raise ValueError("All kappa values must be positive.")
    #     if d < 2:
    #         raise ValueError("Dimension d must be >= 2 for vMF sampling")

    #     # 1. Calculate batch-dependent parameters for Wood/Ulrich rejection sampling
    #     # These now vary per sample based on kappa_batch. Shape: (B,)
    #     kappa_batch_stable = kappa_batch + utils.realmin # Avoid issues with kappa near zero

    #     # Using the formulation from Wood '94 / Ulrich '84 for constants b, x0, c
    #     # sqrt_term = torch.sqrt(4 * kappa_batch_stable**2 + (d - 1)**2)
    #     # b = (d - 1) / (2 * kappa_batch_stable + sqrt_term)
    #     # x0 = (1 - b) / (1 + b)
    #     # c = kappa_batch_stable * x0 + (d - 1) * torch.log(1 - x0**2 + utils.realmin)
    #     # --- Alternative formulation derived from the provided sample_batch code ---
    #     # Let's adapt the constants calculation from the original `sample_batch`
    #     # Note: This `b` definition differs from Wood/Ulrich but was used in the example
    #     kmr = torch.sqrt(4 * kappa_batch_stable**2 + (d - 1)**2)
    #     b_alt = (kmr - 2 * kappa_batch_stable) / (d - 1)
    #     a_alt = (kmr + 2 * kappa_batch_stable + d - 1) / 4
    #     # We need the constants for the acceptance check:
    #     # log(u) <= (d-1)*log(t) - t + dd where t = (2*a*b)/(1-(1-b)*eps)
    #     # This seems more complex to vectorize directly within the loop compared to the
    #     # Wood/Ulrich check: log(u) <= kappa*w + (d-1)*log(1-x0*w) - c
    #     # Let's proceed with the Wood/Ulrich constants (b, x0, c) for clarity and commonality.

    #     sqrt_term = torch.sqrt(4 * kappa_batch_stable**2 + (d - 1)**2)
    #     b = (d - 1) / (2 * kappa_batch_stable + sqrt_term)         # Shape: [B]
    #     x0 = (1 - b) / (1 + b)                                    # Shape: [B]
    #     # Add small epsilon to log argument for stability
    #     c = kappa_batch_stable * x0 + (d - 1) * torch.log(1 - x0**2 + utils.realmin) # Shape: [B]

    #     # Beta distribution for sampling z (candidate for transformation to w)
    #     beta_dist = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))

    #     # 2. Vectorized Rejection Sampling for the radial component 'w'
    #     w_samples = torch.zeros(B, device=device)
    #     accepted_indices = torch.zeros(B, dtype=torch.bool, device=device)
    #     num_accepted = 0

    #     while num_accepted < B:
    #         num_needed = B - num_accepted
    #         # Identify indices that still need samples
    #         unaccepted_mask = ~accepted_indices
    #         current_indices = unaccepted_mask.nonzero(as_tuple=True)[0]

    #         # Sample candidates only for those needed
    #         z_cand = beta_dist.sample((num_needed,)).to(device) # Sample from Beta
    #         u_cand = torch.rand(num_needed, device=device)      # Sample from Uniform

    #         # Get the corresponding kappa, b, x0, c for the samples being processed
    #         kappa_needed = kappa_batch[current_indices]
    #         b_needed = b[current_indices]
    #         x0_needed = x0[current_indices]
    #         c_needed = c[current_indices]

    #         # Calculate w candidates using the transformation (Wood '94 style)
    #         # w = (1 - (1+b)z) / (1 - (1-b)z)
    #         w_cand = (1 - (1 + b_needed) * z_cand) / (1 - (1 - b_needed) * z_cand) # Shape: [num_needed]

    #         # Acceptance condition (vectorized, using per-sample parameters)
    #         # log(u) <= kappa*w + (d-1)*log(1-x0*w) - c
    #         # Add epsilon to log argument
    #         log_term = torch.log(1 - x0_needed * w_cand + utils.realmin)
    #         accept_crit = kappa_needed * w_cand + (d - 1) * log_term - c_needed # Shape: [num_needed]
    #         accept_mask = (torch.log(u_cand + utils.realmin) <= accept_crit) # Shape: [num_needed]

    #         # Find which original indices these accepted candidates correspond to
    #         accepted_indices_in_current = current_indices[accept_mask]

    #         # Store accepted samples
    #         if len(accepted_indices_in_current) > 0:
    #             # Store the corresponding w_cand values
    #             w_samples[accepted_indices_in_current] = w_cand[accept_mask]
    #             accepted_indices[accepted_indices_in_current] = True
    #             num_accepted += len(accepted_indices_in_current)

    #     w = w_samples.unsqueeze(1) # Shape: [B, 1]

    #     # 3. Sample the remaining d-1 dimensions uniformly on the unit sphere S^(d-2)
    #     v_unnorm = torch.randn(B, d - 1, device=device)
    #     v_norm_val = norm(v_unnorm, dim=1, keepdim=True)
    #     # Handle potential zero norm vectors gracefully
    #     v_norm_val = torch.where(v_norm_val == 0, torch.ones_like(v_norm_val), v_norm_val)
    #     v = v_unnorm / v_norm_val # Shape: [B, d-1]

    #     # 4. Combine w and v to form samples aligned with the first basis vector e1
    #     # Ensure w is within [-1, 1] for sqrt; clamp adds numerical stability
    #     w = torch.clamp(w, -1.0 + utils.realmin, 1.0 - utils.realmin)
    #     sqrt_term = torch.sqrt(1 - w**2)
    #     samples_e1 = torch.cat([w, sqrt_term * v], dim=1) # Shape: [B, d]

    #     # 5. Rotate samples from e1 alignment to mu_batch alignment using Householder reflection
    #     e1 = torch.zeros(1, d, device=device)
    #     e1[0, 0] = 1.0 # Shape: [1, d]

    #     u = e1 - mu_batch # Reflection vector, shape: [B, d]
    #     u_norm_val = norm(u, dim=1, keepdim=True)
    #     # Avoid division by zero if mu_batch is already e1
    #     u_norm_val = torch.where(u_norm_val < utils.realmin, torch.ones_like(u_norm_val), u_norm_val)
    #     householder_v = u / u_norm_val # Normalized reflection vector, shape: [B, d]

    #     # Apply reflection: x' = x - 2 * dot(x, v) * v
    #     # Use batch matrix multiplication (bmm) or element-wise product and sum for dot product
    #     # dot_prod = torch.bmm(samples_e1.unsqueeze(1), householder_v.unsqueeze(2)).squeeze() # Needs adjustment
    #     dot_prod = (samples_e1 * householder_v).sum(dim=1, keepdim=True) # Shape: [B, 1]
    #     samples = samples_e1 - 2 * dot_prod * householder_v # Shape: [B, d]

    #     # 6. Final normalization (optional but good practice for numerical stability)
    #     samples = samples / norm(samples, dim=1, keepdim=True)

    #     return samples
    
    @staticmethod
    def sample_batch_mukappa(mu_batch, kappa_batch):
        """
        Samples B vectors efficiently from B different vMF distributions,
        each defined by a pair (mu_i, kappa_i) from the input batches.

        Args:
            mu_batch (torch.Tensor): Batch of mean directions, shape (B, d).
                                     *** Assumed to be normalized (||mu||=1) by the caller. ***
            kappa_batch (torch.Tensor): Batch of concentration parameters, shape (B,) or (B, 1).
                                        Must be non-negative.

        Returns:
            torch.Tensor: Sampled vectors, shape (B, d).

        Notes:
            Operates with no gradient tracking. Uses a vectorized version of the
            Wood (1994) / Ulrich (1984) rejection sampling algorithm.
            Ensures numerical stability for calculations.
        """
        with torch.no_grad():
            # --- Input Validation and Setup ---
            if mu_batch.ndim != 2:
                raise ValueError(f"mu_batch must be a 2D tensor (B, d), got shape {mu_batch.shape}")
            B, d = mu_batch.shape
            if d < 2:
                raise ValueError(f"Dimension d must be >= 2 for vMF sampling, got d={d}")

            # kappa_batch validation and shaping
            if kappa_batch.ndim == 1:
                if kappa_batch.shape[0] != B:
                     raise ValueError(f"Batch size mismatch: mu_batch has B={B}, kappa_batch has B={kappa_batch.shape[0]} (1D)")
                kappa_batch = kappa_batch.unsqueeze(1) # Reshape to (B, 1)
            elif kappa_batch.ndim == 2 and kappa_batch.shape[1] == 1:
                if kappa_batch.shape[0] != B:
                     raise ValueError(f"Batch size mismatch: mu_batch has B={B}, kappa_batch has B={kappa_batch.shape[0]} (2D)")
                pass # Shape is already (B, 1)
            else:
                raise ValueError(f"kappa_batch must be a 1D tensor (B,) or 2D tensor (B, 1), got shape {kappa_batch.shape}")

            # Get device and dtype from inputs
            device = mu_batch.device
            dtype = mu_batch.dtype # Use mu_batch's dtype

            # Ensure kappa values are non-negative for calculations
            if torch.any(kappa_batch < 0):
                 # Raise error instead of just warning - negative kappa is invalid
                 raise ValueError("kappa_batch contains negative values. Kappa must be non-negative.")
            # Clamp to smallest positive value for stability in formulas if kappa is exactly zero
            kappa_batch_safe = torch.clamp(kappa_batch, min=utils.realmin)

            # --- Step 1: Calculate batch parameters for rejection sampling ---
            # Parameters b, x0 (rho), c now depend on kappa_batch, so they become batches.
            d_tensor = torch.tensor(d, device=device, dtype=dtype)
            d_minus_1 = d_tensor - 1.0

            # Calculate b_batch (Wood '94 formula)
            # Note: sqrt( 4k^2 + (d-1)^2 ) is always > 2k for d>=2, k>=0. Denominator > 0.
            sqrt_term_b = torch.sqrt(4 * kappa_batch_safe**2 + d_minus_1**2)
            b_batch = d_minus_1 / (2 * kappa_batch_safe + sqrt_term_b) # Shape: (B, 1)

            # Calculate x0_batch (rho)
            # Denominator 1+b > 0 since b is in [0, 1) for d>=2, k>=0.
            x0_batch = (1 - b_batch) / (1 + b_batch) # Shape: (B, 1)

            # Calculate c_batch = k*x0 + (d-1)*log(1-x0^2)
            # Argument 1-x0^2 = 1 - ((1-b)/(1+b))^2 = ( (1+b)^2 - (1-b)^2 ) / (1+b)^2 = 4b / (1+b)^2
            # Since b >= 0, argument is >= 0. Clamp > tiny for log.
            log_term_arg = 1 - x0_batch**2
            log_term_arg_safe = torch.clamp(log_term_arg, min=utils.realmin)
            log_term_c = torch.log(log_term_arg_safe)
            c_batch = kappa_batch_safe * x0_batch + d_minus_1 * log_term_c # Shape: (B, 1)

            # Distribution for sampling z ~ Beta((d-1)/2, (d-1)/2)
            # Parameter must be positive. d>=2 -> d-1>=1 -> (d-1)/2 >= 0.5
            beta_param = torch.tensor(0.5 * d_minus_1.item(), device=device) # Use scalar tensor param
            beta_dist = torch.distributions.Beta(beta_param, beta_param)

            # --- Step 2: Vectorized Rejection Sampling for 'w' (radial component) ---
            w_samples = torch.zeros(B, 1, device=device, dtype=dtype) # Shape (B, 1)
            # Mask to track which samples have been accepted
            accepted_mask = torch.zeros(B, dtype=torch.bool, device=device)
            num_accepted = 0

            # Keep track of original indices for samples not yet accepted
            original_indices = torch.arange(B, device=device)

            # Add safety break for while loop
            max_iters_rej = B * 1000 # Heuristic limit
            iters_rej = 0

            while num_accepted < B and iters_rej < max_iters_rej:
                iters_rej += 1
                # Determine which samples are still needed
                unaccepted_indices = original_indices[~accepted_mask]
                num_needed = len(unaccepted_indices)

                if num_needed == 0: break # Should not happen if num_accepted < B, but safety

                # Sample candidates ONLY for the needed indices
                z_cand = beta_dist.sample((num_needed,)).to(device) # Shape (num_needed,)
                u_cand = torch.rand(num_needed, device=device, dtype=dtype) # Shape (num_needed,)

                # Get the corresponding parameters for the needed samples by slicing
                # Squeeze result from (num_needed, 1) to (num_needed,) for calculations
                b_needed = b_batch[unaccepted_indices].squeeze(1)
                x0_needed = x0_batch[unaccepted_indices].squeeze(1)
                c_needed = c_batch[unaccepted_indices].squeeze(1)
                kappa_needed = kappa_batch_safe[unaccepted_indices].squeeze(1)

                # Calculate w candidates using the transformation: w = (1 - (1+b)z) / (1 - (1-b)z)
                # Denominator 1-(1-b)z is > 0 since b<1, z<1.
                w_cand = (1 - (1 + b_needed) * z_cand) / (1 - (1 - b_needed) * z_cand) # Shape (num_needed,)

                # Acceptance condition: kappa*w + (d-1)*log(1-x0*w) - c >= log(u)
                # Clamp log arguments > tiny for numerical stability
                log1_x0w_arg = 1 - x0_needed * w_cand
                log1_x0w_arg_safe = torch.clamp(log1_x0w_arg, min=utils.realmin)
                log_u_cand_safe = torch.log(torch.clamp(u_cand, min=utils.realmin))

                accept_crit = kappa_needed * w_cand + d_minus_1 * torch.log(log1_x0w_arg_safe) - c_needed
                current_accept_mask = (accept_crit >= log_u_cand_safe) # Shape (num_needed,) boolean

                # Identify which original indices were accepted in this iteration
                newly_accepted_indices = unaccepted_indices[current_accept_mask]

                if newly_accepted_indices.numel() > 0:
                    # Get the accepted w values corresponding to the original indices
                    accepted_w_values = w_cand[current_accept_mask]

                    # Store the accepted w values in the correct slots of w_samples
                    w_samples[newly_accepted_indices] = accepted_w_values.unsqueeze(1) # Ensure shape (num_newly_accepted, 1)

                    # Update the overall accepted mask
                    accepted_mask[newly_accepted_indices] = True
                    num_accepted = accepted_mask.sum().item() # Update total count

            # Check if loop terminated due to max iterations
            if iters_rej == max_iters_rej and num_accepted < B:
                 unaccepted_count = B - num_accepted
                 print(f"Warning: Rejection sampling reached max iterations ({max_iters_rej}) "
                       f"but only accepted {num_accepted}/{B} samples. "
                       f"{unaccepted_count} samples might be incorrect.")
                 # Decide how to handle this: error, fill with placeholder, or proceed.
                 # Proceeding here, but the unaccepted samples in w_samples will still be 0.

            # w_samples now contains accepted w for all B samples, shape (B, 1)
            w = w_samples

            # --- Step 3: Sample the remaining d-1 dimensions uniformly on S^(d-2) ---
            # Sample standard normal vectors
            v_unnorm = torch.randn(B, d - 1, device=device, dtype=dtype)
            # Normalize each vector v using utils.norm (handles potential zero norms)
            v_norm = utils.norm(v_unnorm, dim=1, keepdim=True) # keepdim=True for division
            v = v_unnorm / v_norm # Shape: [B, d-1]

            # --- Step 4: Combine w and v to form samples aligned with e1 = [1, 0, ..., 0] ---
            # Ensure w is numerically within [-1, 1] for sqrt
            w_clamped = torch.clamp(w, -1.0 + utils.realmin, 1.0 - utils.realmin)
            # Clamp sqrt argument >= 0 just in case of float errors
            sqrt_arg = torch.clamp(1 - w_clamped**2, min=0.0)
            sqrt_term = torch.sqrt(sqrt_arg) # Shape: [B, 1]
            samples_e1 = torch.cat([w_clamped, sqrt_term * v], dim=1) # Shape: [B, d]

            # --- Step 5: Rotate samples from e1 alignment to mu_batch alignment ---
            # Use batched Householder reflection
            e1 = torch.zeros(1, d, device=device, dtype=dtype)
            e1[0, 0] = 1.0 # Shape: [1, d], broadcastable to [B, d]

            # Reflection vectors u = e1 - mu_batch (mu_batch assumed normalized)
            u_reflect = e1 - mu_batch # Shape: [B, d]
            # Calculate norm ||u||. Use keepdim=True for division stability check.
            u_norm = utils.norm(u_reflect, dim=1, keepdim=True) # Shape: [B, 1]

            # Create a mask for samples where mu_batch is not already e1 (needs rotation)
            # Check if norm is significantly larger than zero
            needs_rotation = (u_norm > 1e-6).squeeze(1) # Boolean mask [B]

            # Initialize final samples with the e1-aligned samples
            samples = samples_e1.clone()

            # Only compute reflection for samples that need it
            if needs_rotation.any():
                # Select data for rotation using the boolean mask
                indices_to_rotate = needs_rotation # Alias for clarity
                samples_to_rotate = samples_e1[indices_to_rotate] # Shape [num_rotate, d]
                u_reflect_rotate = u_reflect[indices_to_rotate]   # Shape [num_rotate, d]
                u_norm_rotate = u_norm[indices_to_rotate]         # Shape [num_rotate, 1] (guaranteed > 1e-6 here)

                # Normalized reflection vector v = u / ||u||
                householder_v = u_reflect_rotate / u_norm_rotate # Shape: [num_rotate, d]

                # Apply reflection: x' = x - 2 * dot(x, v) * v
                # Use einsum for robust batch dot product: sum over d for each sample b
                elementwise_prod = samples_to_rotate.view(B, -1) * householder_v.view(B, -1) # Shape [num_rotate, d]
                dot_prod = elementwise_prod.sum(dim=1, keepdim=True) # Shape [num_rotate, 1]

                # Perform the reflection update
                rotated_samples = samples_to_rotate.view(B, -1) - 2 * dot_prod * householder_v.view(B, -1) # Shape: [num_rotate, d]
                
                # Place rotated samples back into the main samples tensor
                samples[indices_to_rotate] = rotated_samples.view(-1)

            # --- Step 6: Final Normalization ---
            # Renormalize samples to ensure they are exactly on the unit sphere
            # due to potential float errors accumulation during rotation.
            final_norms = utils.norm(samples, dim=1, keepdim=True) # keepdim=True for division
            samples = samples / final_norms

            return samples
    

if __name__ == '__main__':
    # Example usage:
    dimension = 3
    mu = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    kappa = 10.0
    num_samples = 5

    samples = sample_vmf(mu, kappa, num_samples)
    print("Samples from von Mises-Fisher distribution:")
    print(samples)

    # Verify that the samples are on the unit sphere (approximately)
    norms = torch.linalg.norm(samples, dim=1)
    print("\nNorms of the samples:")
    print(norms)