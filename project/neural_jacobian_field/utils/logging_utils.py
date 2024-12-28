from dataclasses import is_dataclass

import torch


def get_sanity_metrics(x: dict):
    metrics = {}
    for k, v in x.items():
        # Handle multi-level dictionaries.
        if isinstance(v, dict):
            child_metrics = get_sanity_metrics(v)
            for ck, cv in child_metrics.items():
                metrics[f"{k}_{ck}"] = cv

        # Handle tensors.
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            metrics[f"{k}_max"] = v.max()
            metrics[f"{k}_min"] = v.min()
    return metrics


def safe_asdict(obj):
    """A safe version of asdict that handles PyTorch tensors."""
    if is_dataclass(obj):
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu()  # Convert tensor to numpy array
            else:
                result[key] = safe_asdict(
                    value
                )  # Recursively handle nested dataclasses
        return result
    elif isinstance(obj, dict):
        return {key: safe_asdict(value) for key, value in obj.items()}
    else:
        return obj  # Return the object as is if it's not a dataclass or a tensor
