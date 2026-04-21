import copy

import numpy as np
import torch


def _flatten_state(state_dict):
    """Flatten floating-point tensors in a model state into one vector.

    Why this exists:
        Krum computes pairwise distances between full model updates.
    How it helps:
        Converts structured parameters into a comparable Euclidean representation.
    """
    # Concatenate all floating-point tensors so distances can be computed in one vector space.
    chunks = []
    for tensor in state_dict.values():
        if torch.is_floating_point(tensor):
            chunks.append(tensor.detach().cpu().reshape(-1).float())
    if not chunks:
        raise ValueError("State dict has no floating-point tensors for Krum distance computation")
    return torch.cat(chunks)


def aggregate(local_weights, local_sizes, f=1):
    """Aggregate client updates with Multi-Krum.

    Args:
        local_weights: List of client state_dict updates.
        local_sizes: Unused, kept for API compatibility.
        f: Number of Byzantine clients to tolerate.

    Returns:
        (new_global_state_dict, details)
    """
    # Krum selects updates by geometric consistency rather than dataset size.
    del local_sizes

    num_models = len(local_weights)
    if num_models == 0:
        raise ValueError("local_weights cannot be empty")

    # Need at least 2f + 3 models for Multi-Krum scoring to be well-defined.
    if num_models < (2 * f + 3):
        raise ValueError(f"Krum requires at least 2f+3 clients; got num_models={num_models}, f={f}")

    # Precompute flattened representations for pairwise distance scoring.
    flat_models = [_flatten_state(w) for w in local_weights]

    # Krum score for each client model (lower is better / more central).
    scores = []
    neighbor_count = num_models - f - 2

    # Lower score means a model is closer to many peers (more likely benign).
    for i in range(num_models):
        dists = []
        for j in range(num_models):
            if i == j:
                continue
            diff = flat_models[i] - flat_models[j]
            dists.append(float(torch.dot(diff, diff).item()))

        dists.sort()
        # Sum the nearest neighbor distances, excluding potential outliers.
        scores.append(float(sum(dists[:neighbor_count])))

    # Multi-Krum keeps the best-scoring models and averages only those.
    selected_indices = np.argsort(scores)[: num_models - f]

    new_global = copy.deepcopy(local_weights[0])
    for key in new_global.keys():
        if torch.is_floating_point(new_global[key]):
            # Reset floating tensors before averaging selected model updates.
            new_global[key] = torch.zeros_like(new_global[key])

    for idx in selected_indices:
        state_dict = local_weights[int(idx)]
        for key in new_global.keys():
            if torch.is_floating_point(new_global[key]):
                new_global[key] += state_dict[key]

    denom = float(len(selected_indices))
    for key in new_global.keys():
        if torch.is_floating_point(new_global[key]):
            # Final robust aggregate is a plain mean over selected trustworthy models.
            new_global[key] /= denom

    # Expose selected-client mask in a common weight-vector format for logging.
    adaptive_weights = [0.0] * num_models
    for idx in selected_indices:
        adaptive_weights[int(idx)] = 1.0 / denom

    details = {
        "size_weights": [float("nan")] * num_models,
        "performance_weights": [float("nan")] * num_models,
        "adaptive_weights": adaptive_weights,
    }
    return new_global, details
