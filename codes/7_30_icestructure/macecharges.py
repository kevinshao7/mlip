"""DEPRECATED: MACE-POLAR partial-charge probe.

This script is retained only for historical inspection. Do not use it for new
workflows; current workflows must use formal integer charges only.
"""

raise SystemExit(
    "DEPRECATED: macecharges.py reads MLIP predicted partial charges. "
    "Use formal integer charges only."
)

from mace.calculators import mace_polar

calc = mace_polar(
    model="polar-1-s", # or "polar-1-l"
    device="cpu",           # or "cuda"
    default_dtype="float64" # use float32 for faster MD
)


q = calc.results["charges"]
