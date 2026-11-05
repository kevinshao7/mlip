#!/usr/bin/env python3
"""Task entry point; retain the shared implementation for frozen provenance."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'production'))
sys.path.insert(0, str(ROOT / 'tools'))
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'production/campaign_dashboard.py'), run_name='__main__')
