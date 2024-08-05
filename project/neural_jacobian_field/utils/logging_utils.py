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
