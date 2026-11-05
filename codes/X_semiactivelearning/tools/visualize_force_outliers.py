#!/usr/bin/env python3
"""Compatibility import/command; implementation lives in visualization/."""
import importlib.util
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[1] / 'visualization' / 'visualize_force_outliers.py'
spec = importlib.util.spec_from_file_location(__name__, source)
module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)
