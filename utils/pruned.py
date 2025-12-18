import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

class PCAPrunedSample:
    def __init__(self, vectors: torch.Tensor, shape=[3, 32, 32], dtype=torch.float32, device="cpu", ncandidates: int = 100, threshold: float = 0.12, max_trials: int = 1000):
        """
        vectors: reference tensor of shape (M, D)
        ncandidates: number of proposals per trial
        threshold: max allowed cosine similarity
        max_trials: max number of iterations to attempt
        """
        self.device = device
        self.dtype = dtype
        self.vectors = vectors.to(device=self.device, dtype=self.dtype)
        self.ncandidates = ncandidates
        self.threshold = threshold
        self.max_trials = max_trials
        self.shape = shape

    def sample(self, num_samples: int, threshold: float = None, max_trials: int = None, norm: bool = False, debug: bool = False, reverse: bool = False):
        """
        Sample `num_samples` unit vectors uniformly, rejecting any whose cosine similarity
        with *any* vector in self.vectors exceeds `threshold`.

        Returns:
            Tensor of shape (num_samples, self.shape[0], self.shape[1], self.shape[2])
        """
        if threshold is None:
            threshold = self.threshold
        if max_trials is None:
            max_trials = self.max_trials

        M, D = self.vectors.shape
        matched = torch.empty((num_samples, D), dtype=self.dtype, device=self.device)
        matched_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        trials = 0
        
        if debug:
            sampled = []
            total_candidates_generated = 0
            total_actually_rejected = 0

        while not matched_mask.all() and trials < max_trials:
            needed = (~matched_mask).sum().item()
            pool = needed * self.ncandidates
            
            gauss = torch.randn(pool, D, dtype=self.dtype, device=self.device)
            gauss_norm = F.normalize(gauss, p=2, dim=1)

            sims = gauss_norm @ self.vectors.T
            if reverse:
                valid = sims.ge(threshold).any(dim=1)
            else:
                valid = sims.le(threshold).all(dim=1)
            
            if debug:
                total_candidates_generated += pool
                total_actually_rejected += (pool - valid.sum().item())

            valid_idxs = torch.nonzero(valid, as_tuple=False).squeeze(1)

            if valid_idxs.numel() > 0:
                slots = torch.nonzero(~matched_mask, as_tuple=False).squeeze(1)
                take = min(needed, valid_idxs.numel())
                chosen = valid_idxs[torch.randperm(valid_idxs.numel(), device=self.device)[:take]]
                
                if norm:
                    matched[slots[:take]] = gauss_norm[chosen] * 55.4196
                else:
                    matched[slots[:take]] = gauss[chosen]
                matched_mask[slots[:take]] = True
            
            if debug:
                sampled.append((needed, (matched_mask).sum().item()))
            
            trials += 1

        if debug:
            # MODIFICATION: Calculate the percentage based on "truly" rejected candidates
            if total_candidates_generated > 0:
                pruned_percentage = (total_actually_rejected / total_candidates_generated) * 100
            else:
                pruned_percentage = 0.0

            return sampled, trials, f"Actual Pruned Rate: {pruned_percentage:.2f}%"
        if not matched_mask.all():
            failed = (~matched_mask).sum().item()
            raise RuntimeError(f"Failed to sample {failed} vectors within {max_trials} trials")

        return matched.view(num_samples, self.shape[0], self.shape[1], self.shape[2])

class PrunedSample:
    def __init__(self, ncandidates=1000, threshold=0.08, max_trials=10000):
        self.threshold = threshold
        self.max_trials = max_trials
        self.ncandidates = ncandidates
        self.dataloader = None

    def sample(self, x1, decay_rate=0.666, threshold=None, max_trials=None):
        device = x1.device
        dtype = x1.dtype

        if threshold is None:
            threshold = self.threshold
        if max_trials is None:
            max_trials = self.max_trials

        B, C, H, W = x1.shape
        x1_flat = x1.view(B, -1)
        D = x1_flat.size(1)
        x1_norm = F.normalize(x1_flat, dim=1)

        matched = torch.empty((B, D), dtype=dtype, device=device)
        matched_mask = torch.zeros(B, dtype=torch.bool, device=device)

        trials = 0
        while not matched_mask.all() and trials < max_trials:
            needed = (~matched_mask).sum().item()
            pool_size = needed * self.ncandidates

            gaussian_samples = torch.randn(pool_size, D, dtype=dtype, device=device)
            norm_gaussian = F.normalize(gaussian_samples, dim=1)

            x_unmatched = x1_norm[~matched_mask]        # (needed, D)
            sim = x_unmatched @ norm_gaussian.T         # (needed, pool_size)
            accept = sim > threshold

            unmatched_indices = torch.where(~matched_mask)[0]

            candidates = torch.where(accept.any(dim=0))[0]
            if candidates.numel() >= needed:
                perm = torch.randperm(candidates.size(0), device=device)[:needed]
                chosen = candidates[perm]  # (needed,)
                matched[unmatched_indices] = gaussian_samples[chosen]
                matched_mask[unmatched_indices] = True
            else:
                used = torch.zeros(pool_size, dtype=torch.bool, device=device)
                for i in range(x_unmatched.size(0)):
                    valid = accept[i].nonzero(as_tuple=False).squeeze(1)
                    valid = valid[~used[valid]]
                    if valid.numel() > 0:
                        pick = valid[torch.randint(0, valid.numel(), (1,)).item()]
                        idx = unmatched_indices[i]
                        matched[idx] = gaussian_samples[pick]
                        matched_mask[idx] = True
                        used[pick] = True

            trials += 1

        if not matched_mask.all():
            print(f"[Warning] Failed to find all matches within {max_trials} trials. "
                f"Lowering threshold to {threshold * decay_rate:.4f}")
            return self.sample(
                x1,
                decay_rate=decay_rate,
                threshold=threshold * decay_rate,
                max_trials=max_trials
            )

        return matched.view(-1, C, H, W)
    # replace True
    # def sample(self, x1, decay_rate=0.666, threshold=None, max_trials=None):
    #     device = x1.device
    #     dtype = x1.dtype

    #     if threshold is None:
    #         threshold = self.threshold
    #     if max_trials is None:
    #         max_trials = self.max_trials

    #     B, C, H, W = x1.shape
    #     x1 = x1.view(B, -1)
    #     B, D = x1.shape
    #     x1 = F.normalize(x1, dim=1)
    #     matched = torch.empty((B, D), dtype=dtype, device=device)
    #     matched_mask = torch.zeros(B, dtype=torch.bool, device=device)

    #     trials = 0
    #     while not matched_mask.all() and trials < max_trials:
    #         needed = (~matched_mask).sum().item()
    #         gaussian_samples = torch.randn(needed * self.ncandidates, D, dtype=dtype, device=device)
    #         norm_gaussian_samples = F.normalize(gaussian_samples, dim=1)

    #         x1_unmatched = x1[~matched_mask]
    #         sim = x1_unmatched @ norm_gaussian_samples.T  # (B_unmatched x candidates)
    #         accept_mask = sim > threshold

    #         unmatched_indices = torch.where(~matched_mask)[0]

    #         for i in range(x1_unmatched.size(0)):
    #             valid = accept_mask[i].nonzero(as_tuple=False).squeeze(1)
    #             if valid.numel() > 0:
    #                 # Use softmax over valid similarities
    #                 # weights = torch.softmax(sim[i][valid], dim=0)
    #                 # chosen_idx = valid[torch.multinomial(weights, 1).item()]
    #                 chosen_idx = valid[torch.randint(0, valid.numel(), (1,)).item()]
    #                 matched_idx = unmatched_indices[i]
    #                 matched[matched_idx] = gaussian_samples[chosen_idx]
    #                 matched_mask[matched_idx] = True
    #         trials += 1

    #     if not matched_mask.all():
    #         print(f"[Warning] Failed to find all matches within {max_trials} trials. "
    #             f"Lowering threshold to {threshold * decay_rate:.4f}")
    #         return self.sample(x1, decay_rate=decay_rate, threshold=threshold * decay_rate, max_trials=max_trials)

    #     matched = matched.reshape(-1, 3, 32, 32)
    #     return matched
    # max
    # def sample(self, x1, decay_rate=0.666, threshold=None, max_trials=None):
    #     device = x1.device
    #     dtype = x1.dtype

    #     if threshold is None:
    #         threshold = self.threshold
    #     if max_trials is None:
    #         max_trials = self.max_trials

    #     B, C, H, W = x1.shape
    #     x1 = x1.view(B, -1)
    #     B, D = x1.shape
    #     x1 = F.normalize(x1, dim=1)  # Ensure unit norm
    #     matched = torch.empty((B, D), dtype=dtype, device=device)
    #     matched_mask = torch.zeros(B, dtype=torch.bool, device=device)

    #     trials = 0
    #     while not matched_mask.all() and trials < max_trials:
    #         needed = (~matched_mask).sum().item()
    #         gaussian_samples = torch.randn(needed * self.ncandidates, D, dtype=dtype, device=device)
    #         gaussian_samples = F.normalize(gaussian_samples, dim=1)

    #         # Only consider unmatched x1
    #         x1_unmatched = x1[~matched_mask]
    #         sim = x1_unmatched @ gaussian_samples.T  # (B_unmatched x candidates)

    #         max_sim, max_idx = sim.max(dim=1)  # (B_unmatched,)
    #         accept_mask = max_sim > threshold  # (B_unmatched,)

    #         unmatched_indices = torch.where(~matched_mask)[0]
    #         accepted_indices = unmatched_indices[accept_mask]
    #         accepted_matches = gaussian_samples[max_idx[accept_mask]]

    #         matched[accepted_indices] = accepted_matches
    #         matched_mask[accepted_indices] = True

    #         trials += 1

    #     if not matched_mask.all():
    #         print(f"[Warning] Failed to find all matches within {max_trials} trials. "
    #               f"Lowering threshold to {threshold * decay_rate:.4f}")
    #         return self.sample(x1, decay_rate=decay_rate, threshold=threshold * decay_rate, max_trials=max_trials)
    #     matched = matched.reshape(-1, 3, 32, 32)
        return matched
    def make_loader(self, batch_size):
        BATCH_SIZE = batch_size
        trainset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        self.dataloader = iter(torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True))
        
    def gen_sample(self, batch_size, demo=False, device='cpu'):
        if self.dataloader is None or demo:
            self.make_loader(batch_size)
        x1 = next(self.dataloader)[0].to(device)
        # print("pruned sample", self.threshold)
        return self.sample(x1)

if __name__ == "__main__":
    vectors = torch.load("/data2/projects/junho/conditional-flow-matching/examples/images/cifar10/data/pca_pruned_noflip_under_1e-2.pt")
    sampler = PCAPrunedSample(vectors)
    x = sampler.sample(32)