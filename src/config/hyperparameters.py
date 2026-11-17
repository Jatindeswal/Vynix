#!/usr/bin/env python3
"""
Project Vynix: Frozen Hyperparameters Loader
============================================
Single source of truth for inference hyperparameters.
Reads strictly from frozen_hyperparameters.json to prevent hyperparameter drift.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.config.paths import REPO_ROOT


def get_frozen_hyperparameters_path() -> Path:
    """Finds frozen_hyperparameters.json in review_response or root."""
    candidate_paths = [
        REPO_ROOT / "review_response" / "frozen_hyperparameters.json",
        REPO_ROOT / "frozen_hyperparameters.json",
        REPO_ROOT / "configs" / "frozen_hyperparameters.json",
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find frozen_hyperparameters.json in {[str(p) for p in candidate_paths]}"
    )


def load_all_hyperparameters() -> Dict[str, Any]:
    """Loads the entire frozen hyperparameters dictionary."""
    path = get_frozen_hyperparameters_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_inference_hyperparameters(model_version: str) -> Dict[str, float]:
    """
    Returns frozen inference hyperparameters for model_version ('vynix_1.0' or 'vynix_2.0').
    Keys: 'sigma', 'lambda', 'gamma_obj', 'detector_threshold', 'alpha', 'beta'.
    """
    all_params = load_all_hyperparameters()
    # Normalize version key (e.g., '1.0' -> 'vynix_1.0', 'vynix_1.0' -> 'vynix_1.0')
    normalized_key = model_version if model_version.startswith("vynix_") else f"vynix_{model_version}"
    if normalized_key not in all_params:
        raise KeyError(
            f"Model version '{normalized_key}' not found in frozen_hyperparameters.json. "
            f"Available keys: {[k for k in all_params.keys() if k != 'metadata']}"
        )
    return all_params[normalized_key]
