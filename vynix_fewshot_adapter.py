"""
Compatibility shim for vynix_fewshot_adapter.
Re-exports all symbols from src.models.vynix_fewshot_adapter to preserve 100% backward compatibility.
"""
import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.models.vynix_fewshot_adapter import *
