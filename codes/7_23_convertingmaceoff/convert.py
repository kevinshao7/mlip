import test_torch_to_jax_water_npt as ttj

from pathlib import Path
from types import SimpleNamespace


output_path = Path("macepolarjaxdir").resolve()

def make_output_directory(_name: str) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

tmp_path_factory = SimpleNamespace(
    mktemp=make_output_directory
)

converter = ttj.TestMacePolarPublicConversionAndNPT()

converter.polar_checkpoint(
    tmp_path_factory=tmp_path_factory,
)