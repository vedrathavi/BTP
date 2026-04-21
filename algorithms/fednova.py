import copy

import torch


def aggregate(global_weights, local_weights, local_steps, mu=1.0):
    """Aggregate client updates with FedNova-style normalized deltas.

    Args:
        global_weights: Current global model state_dict.
        local_weights: List of client state_dicts after local training.
        local_steps: List of local optimization step counts per client.
        mu: Server-side scaling factor.

    Returns:
        (new_global_state_dict, details)
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")
    # Step counts must align with client updates to compute normalized contributions.
    if len(local_weights) != len(local_steps):
        raise ValueError("local_weights and local_steps must have the same length")

    total_steps = float(sum(local_steps))
    if total_steps <= 0:
        raise ValueError("Sum of local_steps must be > 0")

    # Normalize each client contribution by the amount of local optimization work.
    norm_weights = [step / total_steps for step in local_steps]

    # Build a normalized update direction from client deltas.
    new_global = copy.deepcopy(global_weights)
    # Update buffer stores aggregated delta before applying server scaling (mu).
    update = {key: torch.zeros_like(value) for key, value in global_weights.items()}

    for weight, state_dict in zip(norm_weights, local_weights):
        for key in global_weights.keys():
            if torch.is_floating_point(global_weights[key]):
                # Delta is measured against the pre-round global model.
                update[key] += float(weight) * (state_dict[key] - global_weights[key])

    for key in new_global.keys():
        if torch.is_floating_point(new_global[key]):
            # Server applies scaled normalized update; mu tunes update aggressiveness.
            new_global[key] = new_global[key] + float(mu) * update[key]

    details = {
        "size_weights": norm_weights,
        "performance_weights": [float("nan")] * len(local_weights),
        "adaptive_weights": norm_weights,
    }
    return new_global, details
