import torch

def stable_coupling_gale_shapley(x0, x1, R0, R1):
    """
    Perform stable coupling using the Gale-Shapley algorithm.

    Args:
        x0 (torch.Tensor): Samples from q0(x0), shape (k, d)
        x1 (torch.Tensor): Samples from q1(x1), shape (k, d)
        R0 (torch.Tensor): Preference rankings of x1 for each x0, shape (k, k)
        R1 (torch.Tensor): Preference rankings of x0 for each x1, shape (k, k)

    Returns:
        torch.Tensor, torch.Tensor: Matched pairs (x0_sampled, x1_sampled)
    """
    k = x0.shape[0]  # Number of samples
    sigma = torch.full((k,), -1, dtype=torch.long)  # Initialize assignments (-1 means unassigned)

    # Track proposal status: which x1 each x0 has already proposed to
    proposals = torch.zeros((k, k), dtype=torch.bool)

    while (sigma == -1).any():  # While there exists an unmatched x0
        for i in range(k):
            if sigma[i] != -1:  # Skip if already matched
                continue

            # Find the first x1 in R0[i] that x0[i] has not proposed to
            for j in R0[i]:
                if not proposals[i, j]:  # Not proposed yet
                    proposals[i, j] = True  # Mark as proposed

                    # Check if x1[j] is already matched
                    i_prime = (sigma == j).nonzero(as_tuple=True)[0]
                    
                    if i_prime.numel() == 0:  # If x1[j] is not matched, assign directly
                        sigma[i] = j
                    else:  # If x1[j] is already matched
                        i_prime = i_prime.item()
                        if R1[j, i] < R1[j, i_prime]:  # If x1[j] prefers x0[i] over current match
                            sigma[i_prime] = -1  # Unmatch previous assignment
                            sigma[i] = j  # Assign new match
                    break  # Move to the next unmatched x0

    # Retrieve matched pairs
    x0_sampled = x0
    x1_sampled = x1[sigma]  # Assign x1 based on sigma

    return x0_sampled, x1_sampled

# Example Usage
k = 10
d = 5  # Dimensionality
x0 = torch.rand(k, d)  # Samples from q0
x1 = torch.rand(k, d)  # Samples from q1

# Generate random rankings
R0 = torch.argsort(torch.rand(k, k), dim=1)  # Preference list for x0
R1 = torch.argsort(torch.rand(k, k), dim=1)  # Preference list for x1

x0_sampled, x1_sampled = stable_coupling_gale_shapley(x0, x1, R0, R1)

print("Sampled x0:", x0_sampled)
print("Sampled x1:", x1_sampled)
