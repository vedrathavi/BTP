import numpy as np
import torch


DEFAULT_CONFIG = {
    "beta_size": 0.6,
    "beta_perf": 0.4,
    "temperature": 2.0,
    "min_client_weight": 1e-3,
    "smoothing_alpha": 0.8,
}


def _init_aggregated_state(reference_state):
    """Create an output state dict template for aggregation.

    Why this exists:
        Aggregation should sum only floating tensors while safely carrying non-float buffers.
    How it helps:
        Prevents type-related issues for entries such as counters or metadata buffers.
    """
    aggregated = {}
    for key, tensor in reference_state.items():
        if torch.is_floating_point(tensor):
            aggregated[key] = torch.zeros_like(tensor)
        else:
            aggregated[key] = tensor.clone()
    return aggregated


def aggregate(local_weights, local_sizes, local_performances, config=None):
    """Aggregate client weights using Adaptive FedAvg.

    What this function does:
        Blends data-size weights and performance-based softmax weights, then averages
        client models with the resulting adaptive coefficients.
    Why it is used:
        Pure size-based averaging may underuse high-quality clients; pure performance
        weighting can be unstable. This combines both signals.
    How it helps:
        Improves robustness under heterogeneous client quality while retaining fairness.

    Config keys:
        - beta_size
        - beta_perf
        - temperature
        - min_client_weight
        - smoothing_alpha
    """
    if not local_weights:
        raise ValueError("local_weights cannot be empty")

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    total_samples = float(sum(local_sizes))
    if total_samples <= 0:
        raise ValueError("Sum of local_sizes must be > 0")

    size_weights = np.array([size / total_samples for size in local_sizes], dtype=np.float64)

    # Clip and standardize local performance so softmax scaling remains stable.
    perf = np.array(local_performances, dtype=np.float64)
    perf = np.clip(perf, 0.1, 0.95)
    perf = (perf - np.mean(perf)) / (np.std(perf) + 1e-6)

    temp = max(float(cfg["temperature"]), 1e-6)
    logits = perf / temp
    logits = logits - np.max(logits)
    perf_scaled = np.exp(logits)
    perf_weights = perf_scaled / np.sum(perf_scaled)

    # Mix size and performance components into one adaptive contribution weight.
    adaptive_weights = (
        float(cfg["beta_size"]) * size_weights
        + float(cfg["beta_perf"]) * perf_weights
    )
    alpha = float(cfg["smoothing_alpha"])
    adaptive_weights = alpha * adaptive_weights + (1.0 - alpha) * size_weights

    # Enforce a minimum contribution and re-normalize.
    adaptive_weights = np.maximum(adaptive_weights, float(cfg["min_client_weight"]))
    adaptive_weights = adaptive_weights / np.sum(adaptive_weights)

    new_global = _init_aggregated_state(local_weights[0])

    for weight, state_dict in zip(adaptive_weights, local_weights):
        for key in state_dict.keys():
            if torch.is_floating_point(state_dict[key]):
                new_global[key] += state_dict[key] * float(weight)

    details = {
        "size_weights": size_weights.tolist(),
        "performance_weights": perf_weights.tolist(),
        "adaptive_weights": adaptive_weights.tolist(),
    }
    return new_global, details
