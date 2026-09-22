import torch


def aggregate(local_weights, local_sizes):
    """Aggregate client states using coordinate-wise median aggregation."""
    del local_sizes

    if not local_weights:
        raise ValueError("local_weights cannot be empty")

    new_global = {}
    for key, reference_tensor in local_weights[0].items():
        if torch.is_floating_point(reference_tensor):
            tensors = [state_dict[key] for state_dict in local_weights]
            stacked = torch.stack(tensors, dim=0)
            sorted_values = torch.sort(stacked, dim=0).values
            middle = sorted_values.shape[0] // 2
            if sorted_values.shape[0] % 2:
                new_global[key] = sorted_values[middle]
            else:
                new_global[key] = (sorted_values[middle - 1] + sorted_values[middle]) / 2
        else:
            new_global[key] = reference_tensor.clone()

    details = {
        "aggregation": "coordinate_wise_median",
        "size_weights": None,
        "performance_weights": None,
        "adaptive_weights": None,
    }
    return new_global, details