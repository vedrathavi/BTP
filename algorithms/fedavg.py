import numpy as np
import torch


MIN_CLIENT_WEIGHT = 1e-12


def _init_aggregated_state(reference_state):
    aggregated = {}
    for key, tensor in reference_state.items():
        if torch.is_floating_point(tensor):
            aggregated[key] = torch.zeros_like(tensor)
        else:
            aggregated[key] = tensor.clone()
    return aggregated


def aggregate(local_weights, local_sizes):
    """Aggregate client weights with classic FedAvg (size-weighted)."""
    if not local_weights:
        raise ValueError("local_weights cannot be empty")

    total_samples = float(sum(local_sizes))
    if total_samples <= 0:
        raise ValueError("Sum of local_sizes must be > 0")

    client_weights = np.array([size / total_samples for size in local_sizes], dtype=np.float64)
    client_weights = np.maximum(client_weights, MIN_CLIENT_WEIGHT)
    client_weights = client_weights / np.sum(client_weights)

    new_global = _init_aggregated_state(local_weights[0])

    for weight, state_dict in zip(client_weights, local_weights):
        for key in state_dict.keys():
            if torch.is_floating_point(state_dict[key]):
                new_global[key] += state_dict[key] * float(weight)

    details = {
        "size_weights": client_weights.tolist(),
        "performance_weights": [float("nan")] * len(local_weights),
        "adaptive_weights": client_weights.tolist(),
    }
    return new_global, details
