import test_torch_to_jax_water_npt as ttj
import torch
from pathlib import Path
converter = ttj.TestMacePolarPublicConversionAndNPT()
converter.test_mace_polar_medium_torch_to_jax_water_npt("MACE-POLAR-1-M.model","./macepolartestdir/")

# path = "MACE-POLAR-1-M.model"
# torch_model = torch.load(
#     path,
#     map_location="cpu",
#     weights_only=False,
# )
# ttj._save_polar_mace_checkpoint(torch_model, Path("macepolarjaxdir").resolve())