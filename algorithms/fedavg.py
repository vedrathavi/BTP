import numpy as np
import torch


MIN_CLIENT_WEIGHT = 1e-12


def _init_aggregated_state(reference_state):
    """Initialize an aggregation target state from a reference model state.

    Why this exists:
        FedAvg sums floating tensors but should preserve non-floating entries safely.
    How it helps:
        Ensures consistent state_dict structure during server-side averaging.
    """
    # This dictionary will mirror the model state_dict schema one-to-one.
    aggregated = {}
    for key, tensor in reference_state.items():
        if torch.is_floating_point(tensor):
            # Start floating parameters from zero because they are accumulated via weighted sum.
            aggregated[key] = torch.zeros_like(tensor)
        else:
            # Copy non-floating buffers as-is (for example, counters) to keep valid state structure.
            aggregated[key] = tensor.clone()
    return aggregated


def aggregate(local_weights, local_sizes):
    """Aggregate client weights with classic FedAvg (size-weighted).

    Why it is used:
        FedAvg is the baseline FL aggregator that weights client updates by data volume.
    How it helps:
        Produces a simple, stable global update under IID or mildly non-IID splits.
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")

    # Total sample count is the denominator for FedAvg's size-based weighting.
    total_samples = float(sum(local_sizes))
    if total_samples <= 0:
        raise ValueError("Sum of local_sizes must be > 0")

    # Convert client sample counts into normalized contribution weights.
    client_weights = np.array([size / total_samples for size in local_sizes], dtype=np.float64)
    # Guard against numerically vanishing clients, then re-normalize to probability simplex.
    client_weights = np.maximum(client_weights, MIN_CLIENT_WEIGHT)
    client_weights = client_weights / np.sum(client_weights)

    # Prepare a zeroed global accumulator compatible with the model structure.
    new_global = _init_aggregated_state(local_weights[0])

    for weight, state_dict in zip(client_weights, local_weights):
        for key in state_dict.keys():
            if torch.is_floating_point(state_dict[key]):
                # Standard FedAvg update: weighted average across client parameters.
                new_global[key] += state_dict[key] * float(weight)

    # Unified metadata shape keeps downstream logging code algorithm-agnostic.
    details = {
        "size_weights": client_weights.tolist(),
        "performance_weights": [float("nan")] * len(local_weights),
        "adaptive_weights": client_weights.tolist(),
    }
    return new_global, details
