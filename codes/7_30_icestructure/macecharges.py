from mace.calculators import mace_polar

calc = mace_polar(
    model="polar-1-s", # or "polar-1-l"
    device="cpu",           # or "cuda"
    default_dtype="float64" # use float32 for faster MD
)


q = calc.results["charges"]
