#!/usr/bin/env python3
"""
Project Vynix: Centralized Path Resolver
=======================================
Resolves paths relative to the repository root using environment variable
overrides. Does NOT contain any hardcoded host-specific drive letters or user paths.
"""

import os
from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """Returns the absolute Path of the repository root."""
    return Path(__file__).resolve().parent.parent.parent


REPO_ROOT = get_repo_root()


def resolve_path(path_str: str) -> str:
    """
    Resolves a path string. If relative, anchors it to REPO_ROOT.
    Expands user and environment variables.
    """
    expanded = os.path.expanduser(os.path.expandvars(path_str))
    p = Path(expanded)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def get_dataset_dir() -> str:
    """Returns the dataset root directory from VYNIX_DATASET_DIR or default."""
    default = str(REPO_ROOT / "data" / "hico_20160224_det")
    return resolve_path(os.environ.get("VYNIX_DATASET_DIR", default))


def get_predictions_dir() -> str:
    """Returns the test predictions directory from VYNIX_PREDICTIONS_DIR or default."""
    default = str(REPO_ROOT / "test_predictions")
    return resolve_path(os.environ.get("VYNIX_PREDICTIONS_DIR", default))


def get_results_dir() -> str:
    """Returns the evaluation results directory from VYNIX_RESULTS_DIR or default."""
    default = str(REPO_ROOT / "results")
    return resolve_path(os.environ.get("VYNIX_RESULTS_DIR", default))


def get_model_dir(model_name: str = "v1.0") -> str:
    """
    Returns the saved model directory for the given version:
    'v1.0', 'v2.0', 'v1.0_train90', or 'v2.0_train90'.
    """
    env_keys = {
        "v1.0": "VYNIX_V1_DIR",
        "v2.0": "VYNIX_V2_DIR",
        "v1.0_train90": "VYNIX_V1_TRAIN90_DIR",
        "v2.0_train90": "VYNIX_V2_TRAIN90_DIR",
    }
    dir_names = {
        "v1.0": "saved_models_v1.0",
        "v2.0": "saved_models_v2.0",
        "v1.0_train90": "saved_models_v1.0_train90",
        "v2.0_train90": "saved_models_v2.0_train90",
    }
    env_var = env_keys.get(model_name, "")
    fallback_dir = dir_names.get(model_name, f"saved_models_{model_name}")
    default = str(REPO_ROOT / fallback_dir)
    return resolve_path(os.environ.get(env_var, default))
