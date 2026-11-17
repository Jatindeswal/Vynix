"""
Compatibility shim for review_response.hico_evaluator.
Re-exports all symbols from src.evaluation.hico_evaluator.
"""
import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.evaluation.hico_evaluator import *
