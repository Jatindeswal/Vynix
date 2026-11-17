#!/usr/bin/env python3
"""
Project Vynix: Unified Checkpoint & Cache Loader
================================================
Single source of truth for loading Vynix checkpoints, caches, and learned parameters.
Enforces strict shape assertions on:
- Cache key dimensions [N, 1536]
- Cache value dimensions [N, 600]
- Spatial MLP layer shapes:
    net.0: [64, 8]
    net.1: [64] (LayerNorm)
    net.3: [600, 64]
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.config.paths import get_model_dir, resolve_path
from src.models.vynix_fewshot_adapter import SpatialMLP


def load_vynix_checkpoint(
    model_dir_or_version: Union[str, Path],
    device: str = "cpu",
    load_to_device: bool = True,
) -> Dict[str, Any]:
    """
    Loads and validates a Vynix checkpoint and cache directory.

    Args:
        model_dir_or_version: Directory path or version string ('v1.0', 'v2.0',
                              'v1.0_train90', 'v2.0_train90').
        device: Device to place tensors and models onto ('cpu', 'cuda', etc.).
        load_to_device: Whether to move tensors to target device immediately.

    Returns:
        Dict containing:
            - 'cache_keys': Tensor [N, 1536]
            - 'cache_values': Tensor [N, 600]
            - 'spatial_mlp': SpatialMLP module in eval() mode
            - 'alpha': float
            - 'beta': float
            - 'num_entries': int (N)
            - 'state_dict': original OrderedDict
    """
    path_str = str(model_dir_or_version)
    if os.path.isdir(path_str):
        model_dir = Path(path_str)
    else:
        resolved = get_model_dir(path_str)
        if os.path.isdir(resolved):
            model_dir = Path(resolved)
        else:
            model_dir = Path(resolve_path(path_str))

    model_pth = model_dir / "vynix_full_model.pth"
    if not model_pth.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_pth}")

    target_device = torch.device(device if load_to_device else "cpu")
    state_dict = torch.load(model_pth, map_location=target_device, weights_only=False)

    # 1. Validate Cache Keys and Values
    assert "cache_keys" in state_dict, f"Missing 'cache_keys' in {model_pth}"
    assert "cache_values" in state_dict, f"Missing 'cache_values' in {model_pth}"

    cache_keys = state_dict["cache_keys"]
    cache_values = state_dict["cache_values"]

    assert cache_keys.ndim == 2, f"Expected 2D cache_keys, got shape {cache_keys.shape}"
    assert cache_keys.shape[1] == 1536, f"Expected cache_keys dimension 1536, got {cache_keys.shape[1]}"

    assert cache_values.ndim == 2, f"Expected 2D cache_values, got shape {cache_values.shape}"
    assert cache_values.shape[1] == 600, f"Expected cache_values dimension 600, got {cache_values.shape[1]}"

    num_entries = cache_keys.shape[0]
    assert cache_values.shape[0] == num_entries, (
        f"Cache keys ({num_entries}) and values ({cache_values.shape[0]}) size mismatch"
    )

    # 2. Validate Learned Alpha and Beta
    assert "alpha" in state_dict, f"Missing 'alpha' in {model_pth}"
    assert "beta" in state_dict, f"Missing 'beta' in {model_pth}"
    alpha = float(state_dict["alpha"].item())
    beta = float(state_dict["beta"].item())

    # 3. Validate Spatial MLP Layer Weights
    expected_mlp_shapes = {
        "spatial_mlp.net.0.weight": torch.Size([64, 8]),
        "spatial_mlp.net.0.bias": torch.Size([64]),
        "spatial_mlp.net.1.weight": torch.Size([64]),
        "spatial_mlp.net.1.bias": torch.Size([64]),
        "spatial_mlp.net.3.weight": torch.Size([600, 64]),
        "spatial_mlp.net.3.bias": torch.Size([600]),
    }
    for key, expected_shape in expected_mlp_shapes.items():
        assert key in state_dict, f"Missing {key} in state_dict of {model_pth}"
        assert state_dict[key].shape == expected_shape, (
            f"Shape mismatch for {key}: expected {expected_shape}, got {state_dict[key].shape}"
        )

    # Reconstruct SpatialMLP instance
    spatial_mlp = SpatialMLP(num_classes=600, hidden_dim=64)
    mlp_state = {
        k.replace("spatial_mlp.", ""): v
        for k, v in state_dict.items()
        if k.startswith("spatial_mlp.")
    }
    spatial_mlp.load_state_dict(mlp_state)
    spatial_mlp.to(target_device)
    spatial_mlp.eval()

    return {
        "cache_keys": cache_keys,
        "cache_values": cache_values,
        "spatial_mlp": spatial_mlp,
        "alpha": alpha,
        "beta": beta,
        "num_entries": num_entries,
        "state_dict": state_dict,
        "model_dir": str(model_dir),
    }
