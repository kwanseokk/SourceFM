import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torch.distributions.chi2 import Chi2
from tqdm import tqdm
import ot
from scipy.stats import wasserstein_distance

# set folder
RUN = 0

def sample_circle(num_samples):
    theta = torch.rand(num_samples) * 2 * torch.pi
    df = 3072
    chi2 = Chi2(df)
    radius = torch.sqrt(chi2.sample((num_samples,)))
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=1)
    return points

def sample_circle_directions(base_angles, num_samples=200, direction_ratios=[0.05, 0.3, 0.65], angle_offset_deg=10.0):
    # assert abs(sum(direction_ratios) - 1.0) < 1e-6, "direction_ratios must sum to 1.0"
    if abs(sum(direction_ratios) - 1.0) > 1e-6:
        counts = direction_ratios
    else:
        num_directions = len(direction_ratios)
        counts = [int(round(r * num_samples)) for r in direction_ratios]
    df = 625
    chi2 = Chi2(df)
    diff = num_samples - sum(counts)
    if diff != 0:
        max_idx = torch.tensor(direction_ratios).argmax().item()
        counts[max_idx] += diff
    angle_offset = torch.deg2rad(torch.tensor(angle_offset_deg))
    all_points = []
    for base_angle, count in zip(base_angles, counts):
        if count == 0:
            continue
        # Add small random perturbation to the angles
        angles = base_angle + torch.linspace(-angle_offset, angle_offset, count)
        random_perturbation = torch.randn(count) * (angle_offset * 0.1)  # 10% of offset as random noise
        angles = angles + random_perturbation
        radius = torch.sqrt(chi2.sample((count,)))
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        points = torch.stack([x, y], dim=1)
        all_points.append(points)
    return torch.cat(all_points, dim=0)

def sample_circle_rejection(num_samples, pca, rejection_threshold=0.9, both=False):
    points = []
    while len(points) < num_samples:
        point = sample_circle(1)
        axis = pca.components_[1]
        axis = torch.tensor(axis).to(device)
        # normalize the pca axis and the point
        if both:
            if torch.abs(torch.dot(point.squeeze(0).to(device) / torch.norm(point), axis / torch.norm(axis))) <= rejection_threshold:
                points.append(point)
        else:
            if - torch.dot(point.squeeze(0).to(device) / torch.norm(point), axis / torch.norm(axis)) <= rejection_threshold:
                points.append(point)
    return torch.cat(points, dim=0)

def calculate_distance_metrics(generated, real, threshold=1.0):
    """Returns: (avg_min_dist, %_samples_above_threshold)"""
    dist_matrix = np.linalg.norm(real[:, None] - generated, axis=2)
    min_dists = np.min(dist_matrix, axis=0)
    return np.mean(min_dists), (min_dists > threshold).mean()*100

def wasserstein_distance(x, y):
    n, m = len(x), len(y)
    a = np.ones(n) / n  # uniform distribution over x
    b = np.ones(m) / m  # uniform distribution over y
    cost_matrix = ot.dist(x, y)  # equivalent to c[i,j] = ||x_i - y_j||_2
    w2_distance = ot.emd2(a, b, cost_matrix)
    return np.sqrt(w2_distance)  # emd2 returns squared distance

def normalized_wasserstein(gen, base_angles):
    radius = 24.989
    x = torch.cos(base_angles) * radius
    y = torch.sin(base_angles) * radius
    base = torch.stack([x, y], dim=1) # shape: (3, 2)

    # for y, find the closest point in base and calculate the proportion of each point
    dist_matrix = np.linalg.norm(base[:, None] - gen, axis=2)  # shape: (3, N_y)
    closest_indices = np.argmin(dist_matrix, axis=0)  # shape: (N_y,)

    # Calculate the proportion of each point in base
    proportions = list(np.bincount(closest_indices, minlength=base.shape[0])) # shape: (3,)

    data = sample_circle_directions(base_angles, num_samples=1024, direction_ratios=proportions)
    data = data.cpu().numpy()  # Convert to numpy for wasserstein_distance

    nw = wasserstein_distance(data, gen)
    return nw

def coverage_precision(generated, real, epsilon=0.5):
    # Precision: % generated near real data
    dist_matrix = np.linalg.norm(real[:, None] - generated, axis=2)
    precision = (np.min(dist_matrix, axis=0) < epsilon).mean()
    
    # Coverage: % real data near generated
    coverage = (np.min(dist_matrix, axis=1) < epsilon).mean()
    return coverage, precision

def coverage_precision_no_replacement(generated, real, epsilon=0.5):
    dist_matrix = np.linalg.norm(real[:, None] - generated, axis=2)  # shape: (N_real, N_gen)

    # Find all pairs within epsilon
    valid_pairs = np.argwhere(dist_matrix < epsilon)

    # Precision: match generated → real (each real used at most once)
    used_real = set()
    used_gen = set()
    precision_matches = 0
    for i, j in valid_pairs:
        if j not in used_gen and i not in used_real:
            used_gen.add(j)
            used_real.add(i)
            precision_matches += 1
    precision = precision_matches / dist_matrix.shape[1]

    # Coverage: match real → generated (each generated used at most once)
    used_real = set()
    used_gen = set()
    coverage_matches = 0
    for i, j in valid_pairs:
        if i not in used_real and j not in used_gen:
            used_real.add(i)
            used_gen.add(j)
            coverage_matches += 1
    coverage = coverage_matches / dist_matrix.shape[0]

    return coverage, precision

def mmd(x, y, sigma=1.0):
    gamma = 1/(2*sigma**2)
    xx = np.exp(-gamma * np.linalg.norm(x[:,None]-x, axis=2)**2)
    yy = np.exp(-gamma * np.linalg.norm(y[:,None]-y, axis=2)**2)
    xy = np.exp(-gamma * np.linalg.norm(x[:,None]-y, axis=2)**2)
    return xx.mean() + yy.mean() - 2*xy.mean()

from sklearn.cluster import KMeans
from scipy.stats import entropy

def compute_mode_distribution_divergence(generated, true_data, num_modes=8):
    # Step 1: Fit KMeans on true data to estimate modes
    kmeans = KMeans(n_clusters=num_modes, n_init='auto').fit(true_data)
    centers = kmeans.cluster_centers_

    # Step 2: Assign samples to nearest mode
    def assign_modes(samples):
        dists = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)  # (N, num_modes)
        assignments = np.argmin(dists, axis=1)
        return assignments

    true_assignments = assign_modes(true_data)
    gen_assignments = assign_modes(generated)

    # Step 3: Compute distributions (normalized histograms)
    true_hist, _ = np.histogram(true_assignments, bins=np.arange(num_modes + 1), density=True)
    gen_hist, _ = np.histogram(gen_assignments, bins=np.arange(num_modes + 1), density=True)

    # Step 4: Compute KL-divergence
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    true_hist = np.clip(true_hist, epsilon, 1)
    gen_hist = np.clip(gen_hist, epsilon, 1)

    mdd = entropy(true_hist, gen_hist)
    return mdd


def plot_trajectories(traj=None, x0=None, title="CFM", pca=None, epsilon=1, threshold=1, sigma=1.0, k='10k', run_idx=0, base_angles=None):
    """Plot trajectories of some selected samples and optionally original x0 points."""
    n = 1024
    plt.figure(figsize=(6, 6))

    if traj is not None:
        # Plot initial Gaussian samples
        plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=6, alpha=0.8, c="black", label="Source")

        if x0 is not None:
            x0 = x0[:n]

        # Loop through each trajectory
        for i in range(n):
            final_point = traj[-1, i]
            if x0 is not None:
                # Compute L2 distance to all x0[:n]
                dists = np.linalg.norm(x0 - final_point, axis=1)
                min_dist = np.min(dists)
            else:
                min_dist = 0  # Default to 0 if x0 is not provided

            color = "lightsteelblue" if min_dist <= 1 else "indianred"
            ss = 1 if min_dist <= 1 else 0.3
            alpha = 0.5 if min_dist <= 1 else 0.2

            # Plot trajectory
            plt.scatter(traj[:, i, 0], traj[:, i, 1], alpha=0.2, color=color, s=ss)

        # Plot final points
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="red", label="Generated")

    if x0 is not None:
        plt.scatter(x0[:, 0], x0[:, 1], s=4, alpha=0.8, c="blue", label="Data")

    # Calculate Matrices
    avg_dist, pct_bad = calculate_distance_metrics(traj[-1], x0, threshold)
    w2 = wasserstein_distance(traj[-1], x0)
    # coverage, precision = coverage_precision(traj[-1], x0, epsilon)
    coverage, precision = coverage_precision_no_replacement(traj[-1], x0, epsilon)
    mmd_score = mmd(traj[-1], x0, sigma)
    mdd = compute_mode_distribution_divergence(traj[-1], x0)
    nw = normalized_wasserstein(traj[-1], base_angles)


    plt.legend(loc='upper left', fontsize=20, markerscale=3)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.tight_layout()
    # plt.show()
    # make dir if not exists
    os.makedirs(f"results_{RUN}/{title}/{run_idx}", exist_ok=True)
    plt.savefig(f"results_{RUN}/{title}/{run_idx}/{title}_{k}.png", dpi=300)
    plt.close()

    return [avg_dist, pct_bad, w2, coverage, mmd_score, mdd, nw]

def angle_to_vector(angle):
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

def sample_circle_rejection_dir_ratio(base_angles, direction_ratios, num_samples=200, rejection_threshold=0.9, batch_size=128):
    # base_angles_tensor = torch.tensor(base_angles)
    base_vectors = angle_to_vector(base_angles)  # shape: [3, 2]

    # Determine sample counts per direction
    sample_counts = [int(r * num_samples) for r in direction_ratios]

    # Adjust in case rounding drops samples
    remainder = num_samples - sum(sample_counts)
    for i in range(remainder):
        sample_counts[i % len(sample_counts)] += 1

    accepted_all = []

    for i in range(len(base_vectors)):
        base_vec = base_vectors[i]  # shape: [2]
        count_needed = sample_counts[i]
        accepted = []

        while len(accepted) < count_needed:
            batch_points = sample_circle(batch_size)  # [B, 2]
            directions = batch_points / batch_points.norm(dim=1, keepdim=True)  # [B, 2]

            # Cosine similarity with base_vec
            cos_sim = torch.matmul(directions, base_vec)  # [B]

            # Accept points with high similarity
            mask = cos_sim >= rejection_threshold
            accepted_batch = batch_points[mask]

            accepted.append(accepted_batch)

        accepted_dir = torch.cat(accepted, dim=0)[:count_needed]
        accepted_all.append(accepted_dir)

    return torch.cat(accepted_all, dim=0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sigma = 0
    dim = 2
    batch_size = 10
    total_samples = 1024

    torch.manual_seed(124142)

    base_angles = torch.rand(3) * 2 * torch.pi
    # x1 sampled
    fixed_x1 = sample_circle_directions(base_angles, total_samples).to(device)

    # Calculate PCA axis from fixed_x1
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(fixed_x1.cpu().numpy())

    model = MLP(dim=dim, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Store results
    n_models = 10
    n_runs = 10
    n_metrics = 7   # avg_min_dist, pct_samples_above_threshold, w2, coverage, mmd_score, mdd
    n_checkpoints = 10 # (10k, 20k, ..., 100k)

    results = np.zeros((n_models, n_runs, n_checkpoints, n_metrics))

    def log_results(model_id, run_id, iter_idx, metrics):
        """Log results for a specific model, run, and iteration."""
        results[model_id, run_id, iter_idx, :] = metrics

    # 1. OT-CFM
    model_id = 0
    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for OT-CFM")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        # Initialize the ExactOptimalTransportConditionalFlowMatcher
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        start = time.time()
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]
            x0 = sample_circle(batch_size).to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                # end = time.time()
                # print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                # start = end
                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )
                
                with torch.no_grad():
                    traj = node.trajectory(
                        sample_circle(1024).to(device),
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="OT-CFM", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 2. CFM
    model_id = 1
    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        # Initialize the ConditionalFlowMatcher
        FM = ConditionalFlowMatcher(sigma=sigma)
        start = time.time()
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]
            x0 = sample_circle(batch_size).to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                # end = time.time()
                # print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                # start = end
                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        sample_circle(1024).to(device),
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="CFM", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)
    
    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 3. CFM with rejection sampling
    model_id = 2
    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Rej")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ConditionalFlowMatcher(sigma=sigma)
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]
            # Use rejection sampling to sample x0
            # x0 = sample_circle_rejection(batch_size, pca, rejection_threshold=0.98).to(device)
            x0 = sample_circle(batch_size).to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)
            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                # end = time.time()
                # print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                # start = end
                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        sample_circle_rejection(1024, pca, rejection_threshold=0.98).to(device),
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="CFM-Rej", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 4. CFM iter 20
    model_id = 3
    base_model = MLP(dim=dim, time_varying=True).to(device)
    base_model.load_state_dict(torch.load('viz_models/base_model_20iter.pth'))  # Copy initial weights

    static_node = NeuralODE(
        torch_wrapper(base_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    with torch.no_grad():
        samples = static_node.trajectory(
                sample_circle(20000).to(device),
                t_span=torch.linspace(0, 1, 100).to(device),
            )
        
    samples = samples[-1, :].to(device)
    # samples = samples / torch.mean(torch.norm(samples, dim=-1, keepdim=True)) * 55.4185
    
    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Iter20")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]

            # random sample from traj to use as x0 from 20000 samples
            random_indices = torch.randint(0, samples.shape[0], (batch_size,))
            
            x0 = samples[random_indices].to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                random_indices = torch.randint(0, samples.shape[0], (1024,))
                noise = samples[random_indices].to(device)

                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        noise,
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="CFM-Iter20", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)


    # 5. CFM iter 200
    model_id = 4
    base_model = MLP(dim=dim, time_varying=True).to(device)
    base_model.load_state_dict(torch.load('viz_models/base_model_200iter.pth'))  # Copy initial weights

    static_node = NeuralODE(
        torch_wrapper(base_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )   
    with torch.no_grad():
        samples = static_node.trajectory(
                sample_circle(20000).to(device),
                t_span=torch.linspace(0, 1, 100).to(device),
        )
    samples = samples[-1, :].to(device)
    # samples = samples / torch.mean(torch.norm(samples, dim=-1, keepdim=True)) * 55.4185

    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Iter200")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]

            # random sample from traj to use as x0 from 20000 samples
            random_indices = torch.randint(0, samples.shape[0], (batch_size,))

            x0 = samples[random_indices].to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                random_indices = torch.randint(0, samples.shape[0], (1024,))
                noise = samples[random_indices].to(device)

                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        noise,
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="CFM-Iter200", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 6. CFM iter 10000
    model_id = 5
    base_model = MLP(dim=dim, time_varying=True).to(device)
    base_model.load_state_dict(torch.load('viz_models/base_model_10000iter.pth'))  # Copy initial weights

    static_node = NeuralODE(
        torch_wrapper(base_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        samples = static_node.trajectory(
                sample_circle(20000).to(device),
                t_span=torch.linspace(0, 1, 100).to(device),
        )
    samples = samples[-1, :].to(device)
    # samples = samples / torch.mean(torch.norm(samples, dim=-1, keepdim=True)) * 55.4185

    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Iter10000")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]

            # random sample from traj to use as x0 from 20000 samples
            random_indices = torch.randint(0, samples.shape[0], (batch_size,))

            x0 = samples[random_indices].to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                random_indices = torch.randint(0, samples.shape[0], (1024,))
                noise = samples[random_indices].to(device)

                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        noise,
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="CFM-Iter10000", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 7. CFM Dir source with ICFM
    direction_ratios = [0.05, 0.3, 0.65]
    model_id = 6

    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Dir-ICFM")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]
            x0 = sample_circle_rejection_dir_ratio(base_angles, direction_ratios, batch_size, rejection_threshold=0.9).to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        sample_circle_rejection_dir_ratio(base_angles, direction_ratios, 1024, rejection_threshold=0.9).to(device),
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="DirICFM", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 8. CFM Dir source with Perfect Pairing
    model_id = 7
    direction_ratios = [0.05, 0.3, 0.65]

    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Dir-ICFM")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        batch_size = 1024

        for k in tqdm(range(3125)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]
            x0 = sample_circle_rejection_dir_ratio(base_angles, direction_ratios, batch_size, rejection_threshold=0.9).to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 312 == 0:
                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        sample_circle_rejection_dir_ratio(base_angles, direction_ratios, 1024, rejection_threshold=0.9).to(device),
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="DirPerfectPairing", k=f"{(k+1)//312}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 312 - 1, metrics)

        # Save results to file
        np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 9. CFM Dir source with ICFM
    direction_ratios = [0.05, 0.3, 0.65]
    model_id = 8

    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Dir-ICFM-Tight")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]
            x0 = sample_circle_rejection_dir_ratio(base_angles, direction_ratios, batch_size, rejection_threshold=0.98).to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        sample_circle_rejection_dir_ratio(base_angles, direction_ratios, 1024, rejection_threshold=0.98).to(device),
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="DirICFM-Tight", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # 10. CFM iter 6000
    model_id = 9
    base_model = MLP(dim=dim, time_varying=True).to(device)
    base_model.load_state_dict(torch.load('viz_models/base_model_6000iter.pth'))  # Copy initial weights

    static_node = NeuralODE(
        torch_wrapper(base_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        samples = static_node.trajectory(
                sample_circle(20000).to(device),
                t_span=torch.linspace(0, 1, 100).to(device),
        )
    samples = samples[-1, :].to(device)
    # samples = samples / torch.mean(torch.norm(samples, dim=-1, keepdim=True)) * 55.4185

    for run_id in range(n_runs):
        print(f"Run {run_id+1}/{n_runs} for CFM-Iter10000")
        model = MLP(dim=dim, time_varying=True).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        for k in tqdm(range(100000)):
            optimizer.zero_grad()

            # Sample batch indices from fixed_x1
            indices = torch.randperm(total_samples)[:batch_size]

            # random sample from traj to use as x0 from 20000 samples
            random_indices = torch.randint(0, samples.shape[0], (batch_size,))

            x0 = samples[random_indices].to(device)
            x1 = fixed_x1[indices].to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            if (k + 1) % 10000 == 0:
                random_indices = torch.randint(0, samples.shape[0], (1024,))
                noise = samples[random_indices].to(device)

                node = NeuralODE(
                    torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
                )

                with torch.no_grad():
                    traj = node.trajectory(
                        noise,
                        t_span=torch.linspace(0, 1, 100).to(device),
                    )
                metrics = plot_trajectories(traj.cpu().numpy(), fixed_x1.cpu().numpy(), title="CFM-Iter6000", k=f"{(k+1)//1000}k", run_idx=run_id, base_angles=base_angles)
                log_results(model_id, run_id, (k+1) // 10000 - 1, metrics)

    # Save results to file
    np.save(f"results_{RUN}/metrics_results_{model_id}.npy", results)

    # Plotting

    model_names = [
        "OT-CFM",
        "ICFM",
        "ICFM-Rej",
        "ICFM-Iter20",
        "ICFM-Iter200",
        "ICFM-Iter10000",
        "ICFM-Dir",
        "ICFM-Dir-PerfectPairing",
        "ICFM-Iter6000"
    ]

    metrics_names = [
        "Avg Min Dist",
        "% Samples > 1.0",
        "Wasserstein Distance",
        "Coverage",
        "MMD",
        "MDD",
        "Normalized Wasserstein"
    ]

    # os.makedirs(f"results_{RUN}/plots", exist_ok=True)

    # for i, metric_name in enumerate(metrics_names):
    #     plt.figure(figsize=(10, 6))
    #     for j in range(n_models):
    #         means = results[j, :, :, i].mean(axis=0)
    #         stds = results[j, :, :, i].std(axis=0)
    #         plt.plot(np.arange(1, n_checkpoints + 1) * 10, means, label=model_names[j])
    #         plt.fill_between(np.arange(1, n_checkpoints + 1) * 10, means - stds, means + stds, alpha=0.2)

    #     plt.title(f"{metric_name} over Training Steps")
    #     plt.xlabel("Training Steps (x1000)")
    #     plt.ylabel(metric_name)
    #     plt.xticks(np.arange(0, n_checkpoints + 1) * 10)
    #     plt.legend()
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.savefig(f"results_{RUN}/plots/{metric_name.replace(' ', '_')}.png", dpi=300)
    #     plt.close()