import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _init_aggregated_state(reference_state):
    aggregated = {}
    for key, tensor in reference_state.items():
        if torch.is_floating_point(tensor):
            aggregated[key] = torch.zeros_like(tensor)
        else:
            aggregated[key] = tensor.clone()
    return aggregated


def aggregate(local_weights, local_sizes):
    """Aggregate client updates with FedMA-style filter matching for conv weights.

    Args:
        local_weights: List of client state_dict updates.
        local_sizes: Unused, kept for API compatibility.

    Returns:
        (new_global_state_dict, details)
    """
    del local_sizes

    if not local_weights:
        raise ValueError("local_weights cannot be empty")

    num_clients = len(local_weights)
    new_global = _init_aggregated_state(local_weights[0])

    for key in new_global.keys():
        ref_tensor = local_weights[0][key]

        if not torch.is_floating_point(ref_tensor):
            continue

        # Apply Hungarian matching only on convolution kernels.
        if key.endswith("weight") and ref_tensor.ndim == 4:
            base = local_weights[0][key]
            base_flat = base.reshape(base.size(0), -1)
            aligned = [base]

            for client_idx in range(1, num_clients):
                target = local_weights[client_idx][key]
                target_flat = target.reshape(target.size(0), -1)

                sim = F.cosine_similarity(
                    base_flat.unsqueeze(1),
                    target_flat.unsqueeze(0),
                    dim=2,
                )
                cost = -sim.detach().cpu().numpy()
                _, col_ind = linear_sum_assignment(cost)
                col_ind = torch.as_tensor(col_ind, device=target.device, dtype=torch.long)
                aligned.append(target.index_select(0, col_ind))

            new_global[key] = torch.mean(torch.stack(aligned, dim=0), dim=0)
        else:
            tensors = [state_dict[key] for state_dict in local_weights]
            new_global[key] = torch.mean(torch.stack(tensors, dim=0), dim=0)

    uniform = [1.0 / num_clients] * num_clients
    details = {
        "size_weights": [float("nan")] * num_clients,
        "performance_weights": [float("nan")] * num_clients,
        "adaptive_weights": uniform,
    }
    return new_global, details
