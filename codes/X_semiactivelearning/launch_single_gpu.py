#!/usr/bin/env python3
"""Stable command entry point; implementation lives in production/."""
import importlib.util
from pathlib import Path
import sys

source = Path(__file__).resolve().parent / "production" / Path(__file__).name
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location(__name__, source)
module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)
