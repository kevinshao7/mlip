import test_torch_to_jax_water_npt as ttj
import torch
from pathlib import Path
path = "MACE-OFF24_medium.model"
torch_model = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)
ttj._save_mace_off_checkpoint(torch_model, Path("jaxdir").resolve())