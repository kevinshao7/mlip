"""End-to-end tests: download a public torch MACE model, convert to JAX, run MD water.

Validates both the mace_off and mace_polar code paths after the dependency swap from
private Angstrom-AI forks to public ACEsuit/mace + graph_electrostatics.

- MACE-OFF: NPT water, density check (float64)
- MACE-POLAR-1-M: NPT water, density check (float32; matches the production config
  in example_configs/mace_polar_npt.toml and the reference April-2026 production run)
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path

import numpy as np
import pytest

# e3nn ≥ 0.5 and torch ≥ 2.6 changed the torch.load default. The env var
# forces weights_only=False, which is required to unpickle mace .model files.
# This must be set before any e3nn or mace import.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_mace_off_checkpoint(torch_model, checkpoint_dir: Path) -> None:
    """Convert a torch MACE-OFF model to JAX and save as an Orbax checkpoint."""
    import orbax.checkpoint as ocp
    from aai_mace.torchmace.conversion.flax_from_torch import load_from_torch
    from aai_mace.mace_jax.models.mace_off import MACE

    _, params, extra_config, mace_flax_config = load_from_torch(torch_model)

    flax_model = MACE(**mace_flax_config)
    config_to_save = flax_model.config.copy()

    atomic_numbers = extra_config.get("atomic_numbers")
    if atomic_numbers is not None:
        an = list(atomic_numbers) if not isinstance(atomic_numbers, list) else atomic_numbers
        config_to_save["atomic_numbers"] = an
        if config_to_save.get("atom_energies") is not None:
            ae = config_to_save["atom_energies"]
            ae = ae.tolist() if hasattr(ae, "tolist") else list(ae)
            config_to_save["atom_energies"] = {str(z): e for z, e in zip(an, ae)}

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "model_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    # options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    # ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=options)
    # ckpt_mgr.save(1, args=ocp.args.StandardSave(params))

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        create=True,
        cleanup_tmp_directories=True,
        enable_async_checkpointing=False,
    )

    with ocp.CheckpointManager(checkpoint_dir, options=options) as ckpt_mgr:
        saved = ckpt_mgr.save(
            1,
            args=ocp.args.StandardSave(params),
        )

        if not saved:
            raise RuntimeError("Orbax did not perform the checkpoint save.")

        print(f"Checkpoint saved to {checkpoint_dir / '1'}")

        ckpt_mgr.wait_until_finished()


def _save_polar_mace_checkpoint(torch_model, checkpoint_dir: Path,
                                 dtype: str = "float32") -> None:
    """Convert a torch PolarMACE model to JAX and save as an Orbax checkpoint.

    dtype="float32" keeps the native torch params (production default).
    dtype="float64" upcasts params and uses HIGHEST TP precision so float64 leaves
    don't get downcast to float32 in TensorProduct accumulators.
    """
    import jax
    import jax.numpy as jnp
    from aai_mace.torchmace.conversion.flax_from_torch import load_from_torch
    from aai_mace.mace_jax.save_load_utils import save_polar_mace_flax_to_orbax
    from aai_mace.mace_jax.models.mace_fukui import PolarMACE

    if dtype == "float64":
        jax.config.update('jax_enable_x64', True)

    _, params, extra_config, mace_flax_config = load_from_torch(torch_model)

    if dtype == "float64":
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        )

    # Convert tuple fields (they come back from config_from_mace_fukui_torch as tuples)
    for field in ("tp_widths", "mlp_hidden", "readout_mlp_hidden",
                  "field_feature_widths", "field_feature_per_l_norms"):
        if field in mace_flax_config and mace_flax_config[field] is not None:
            mace_flax_config[field] = tuple(mace_flax_config[field])

    for _prec_key in ("tp_precision", "contraction_precision", "linear_precision", "mlp_precision"):
        mace_flax_config.pop(_prec_key, None)

    precision_kwargs = (
        {"tp_precision": jax.lax.Precision.HIGHEST,
         "contraction_precision": jax.lax.Precision.HIGHEST}
        if dtype == "float64" else {}
    )
    flax_model = PolarMACE(**mace_flax_config, **precision_kwargs)
    atomic_numbers = np.array(extra_config.get("atomic_numbers", []))

    save_polar_mace_flax_to_orbax(
        params,
        flax_model,
        str(checkpoint_dir),
        atomic_numbers if len(atomic_numbers) > 0 else None,
        extra_config=extra_config,
    )


def _build_water_npt_config(checkpoint_path: str, model_type: str, output_dir: Path,
                             steps: int = 2000, dtype: str = "float64") -> "config.MDConfig":
    """Return an MDConfig for a short NH-NPT water simulation."""
    from jax_md_cli import config

    md = config.MDConfig()
    md.simulation.seed = 42
    md.simulation.dtype = dtype
    md.simulation.steps = steps
    md.simulation.timestep = 0.5  # fs — required for NPT water stability

    # Generate a ~20 Å water box at experimental density
    md.structure.generate.enabled = True
    md.structure.generate.solvent_smiles = "O"
    md.structure.generate.solute_smiles = "O"
    md.structure.generate.density = 0.997  # g/cm³
    md.structure.generate.padding = 10.0   # box = 2*padding = 20 Å
    md.structure.generate.tolerance = 2.0
    md.structure.generate.seed = 42

    md.ensemble.type = "npt"
    md.ensemble.temperature = 300.0   # K
    md.ensemble.pressure = 1.0        # bar
    md.ensemble.barostat_tau = 5.0    # ps — slow coupling avoids overshoot
    md.ensemble.thermostat_tau = 0.5  # ps

    md.potential.type = "nn"
    md.potential.nn.checkpoint_path = checkpoint_path
    md.potential.nn.model_type = model_type
    if model_type == "mace_polar":
        md.potential.nn.total_charge = 0.0
        md.potential.nn.total_spin = 0.0

    md.minimization.enabled = True
    md.minimization.max_steps = 1000
    md.minimization.force_tolerance = 0.001  # eV/Å

    md.output.directory = str(output_dir)
    md.output.log_interval = 200
    md.output.trajectory_interval = 1000
    md.output.overwrite = True

    return md


def _build_water_mc_barostat_config(checkpoint_path: str, model_type: str, output_dir: Path,
                                     steps: int = 4000, dtype: str = "float32") -> "config.MDConfig":
    """Return an MDConfig for NPT water using MC barostat (matches production config).

    MC barostat (nvt_mc_barostat) is the production ensemble for MACE-POLAR: volume
    changes are accepted/rejected via Metropolis criterion on energy differences, so
    no JVP through kspace electrostatics is required for pressure estimation.
    """
    from jax_md_cli import config

    md = config.MDConfig()
    md.simulation.seed = 42
    md.simulation.dtype = dtype
    md.simulation.steps = steps
    md.simulation.timestep = 1.0  # fs — 1 fs is stable for MC barostat

    # Generate a ~20 Å water box at experimental density
    md.structure.generate.enabled = True
    md.structure.generate.solvent_smiles = "O"
    md.structure.generate.solute_smiles = "O"
    md.structure.generate.density = 0.997  # g/cm³
    md.structure.generate.padding = 10.0   # box = 2*padding = 20 Å
    md.structure.generate.tolerance = 2.0
    md.structure.generate.seed = 42

    md.ensemble.type = "nvt_mc_barostat"
    md.ensemble.temperature = 300.0   # K
    md.ensemble.pressure = 1.0        # bar
    md.ensemble.gamma = 1.0           # Langevin friction, 1/ps

    md.potential.type = "nn"
    md.potential.nn.checkpoint_path = checkpoint_path
    md.potential.nn.model_type = model_type
    if model_type == "mace_polar":
        md.potential.nn.total_charge = 0.0
        md.potential.nn.total_spin = 0.0

    md.minimization.enabled = True
    md.minimization.max_steps = 1000
    md.minimization.force_tolerance = 0.001  # eV/Å — tight enough to avoid residual-force temperature spikes

    md.output.directory = str(output_dir)
    md.output.log_interval = 200
    md.output.trajectory_interval = 1000
    md.output.overwrite = True

    return md


def _build_water_nvt_config(checkpoint_path: str, model_type: str, output_dir: Path,
                             steps: int = 1000, dtype: str = "float32") -> "config.MDConfig":
    """Return an MDConfig for a short NVT water simulation."""
    from jax_md_cli import config

    md = config.MDConfig()
    md.simulation.seed = 42
    md.simulation.dtype = dtype
    md.simulation.steps = steps
    md.simulation.timestep = 0.5  # fs

    md.structure.generate.enabled = True
    md.structure.generate.solvent_smiles = "O"
    md.structure.generate.solute_smiles = "O"
    md.structure.generate.density = 0.997  # g/cm³
    md.structure.generate.padding = 10.0   # box = 2*padding = 20 Å
    md.structure.generate.tolerance = 2.0
    md.structure.generate.seed = 42

    md.ensemble.type = "nvt_langevin"
    md.ensemble.temperature = 300.0  # K
    md.ensemble.gamma = 1.0  # friction coefficient, 1/ps

    md.potential.type = "nn"
    md.potential.nn.checkpoint_path = checkpoint_path
    md.potential.nn.model_type = model_type
    if model_type == "mace_polar":
        md.potential.nn.total_charge = 0.0
        md.potential.nn.total_spin = 0.0

    md.minimization.enabled = True
    md.minimization.max_steps = 1000
    md.minimization.force_tolerance = 0.001  # eV/Å

    md.output.directory = str(output_dir)
    md.output.log_interval = 200
    md.output.trajectory_interval = 500
    md.output.overwrite = True

    return md


def _mean_density_last_half(thermo_csv: Path) -> float:
    """Return the mean density (g/cm³) over the second half of a thermo.csv."""
    rows = []
    with open(thermo_csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            density_str = parts[8].strip()
            if density_str:
                try:
                    rows.append(float(density_str))
                except ValueError:
                    pass
    assert len(rows) > 0, "No density values found in thermo.csv"
    rows_arr = np.array(rows, dtype=float)
    finite = rows_arr[np.isfinite(rows_arr)]
    assert len(finite) > 0, "All density values are NaN/inf — simulation exploded"
    half = len(finite) // 2
    return float(np.mean(finite[half:]))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestMaceOffPublicConversionAndNPT:
    """Download public MACE-OFF medium torch model, convert, run NPT water."""

    def test_mace_off_medium_torch_to_jax_water_npt(self, tmp_path):
        from mace.calculators import mace_off
        from jax_md_cli import runner

        # 1. Download and load the public MACE-OFF medium torch model
        print("Downloading MACE-OFF medium torch model...")
        torch_model = mace_off(model="medium", device="cpu", return_raw_model=True)
        print(f"Loaded torch model: {type(torch_model).__name__}")

        # 2. Convert torch → JAX and save to a temp Orbax checkpoint
        checkpoint_dir = tmp_path / "mace_off_medium_jax"
        print(f"Converting to JAX checkpoint at {checkpoint_dir}...")
        _save_mace_off_checkpoint(torch_model, checkpoint_dir)
        assert (checkpoint_dir / "model_config.json").exists()

        # 3. Run a short NPT water simulation with the freshly converted model
        output_dir = tmp_path / "run_water_npt_mace_off"
        md_config = _build_water_npt_config(
            str(checkpoint_dir), "mace_off", output_dir, steps=2000
        )
        print("Running NPT water simulation (mace_off)...")
        runner.run_simulation(md_config)

        # 4. Check water density is close to 1 g/cm³
        thermo_csv = output_dir / "thermo.csv"
        assert thermo_csv.exists(), "thermo.csv not produced"
        mean_rho = _mean_density_last_half(thermo_csv)
        print(f"Mean water density (MACE-OFF medium): {mean_rho:.4f} g/cm³")
        assert 0.85 < mean_rho < 1.10, (
            f"MACE-OFF water density {mean_rho:.3f} g/cm³ outside expected range [0.85, 1.10]"
        )


class TestMacePolarPublicConversionAndNPT:
    """Download public MACE-POLAR-1-M torch model, convert, run NPT water.

    Uses MC barostat (nvt_mc_barostat) with float32 — the same configuration as the
    production MACE-POLAR config (example_configs/mace_polar_npt.toml).
    """

    # @pytest.fixture(scope="class")
    def polar_checkpoint(self, tmp_path_factory):
        """Convert the torch model once and share across tests in this class."""
        from mace.calculators import mace_polar

        tmp_path = tmp_path_factory.mktemp("polar_ckpt")
        print("Downloading MACE-POLAR-1-M torch model...")
        torch_model = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)
        print(f"Loaded torch model: {type(torch_model).__name__}")

        checkpoint_dir = tmp_path / "mace_polar_medium_jax_f32"
        print(f"Converting to JAX float32 checkpoint at {checkpoint_dir}...")
        _save_polar_mace_checkpoint(torch_model, checkpoint_dir, dtype="float32")
        assert (checkpoint_dir / "model_config.json").exists()
        return checkpoint_dir

    # @pytest.fixture(scope="class")
    def polar_checkpoint_f64(self, tmp_path_factory):
        """Convert the torch model to float64 once and share across tests."""
        from mace.calculators import mace_polar

        tmp_path = tmp_path_factory.mktemp("polar_ckpt_f64")
        print("Downloading MACE-POLAR-1-M torch model (float64 conversion)...")
        torch_model = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)

        checkpoint_dir = tmp_path / "mace_polar_medium_jax_f64"
        print(f"Converting to JAX float64 checkpoint at {checkpoint_dir}...")
        _save_polar_mace_checkpoint(torch_model, checkpoint_dir, dtype="float64")
        assert (checkpoint_dir / "model_config.json").exists()
        return checkpoint_dir

    def test_mace_polar_medium_torch_to_jax_water_npt(self, polar_checkpoint, tmp_path):
        """NPT water simulation using MC barostat (production-matching config, float32).

        MC barostat (nvt_mc_barostat) is the production ensemble for MACE-POLAR: it
        proposes volume moves and accepts/rejects on energy differences, so no JVP
        through kspace electrostatics is needed. Float32 matches the production config.
        """
        from jax_md_cli import runner

        output_dir = tmp_path / "run_water_npt_mace_polar_mc"
        md_config = _build_water_mc_barostat_config(
            str(polar_checkpoint), "mace_polar", output_dir, steps=4000
        )
        print("Running NPT water simulation with MC barostat (mace_polar, float32)...")
        runner.run_simulation(md_config)

        thermo_csv = output_dir / "thermo.csv"
        assert thermo_csv.exists(), "thermo.csv not produced"
        mean_rho = _mean_density_last_half(thermo_csv)
        print(f"Mean water density (MACE-POLAR-1-M, MC barostat, float32): {mean_rho:.4f} g/cm³")
        assert 0.80 < mean_rho < 1.15, (
            f"MACE-POLAR-1-M water density {mean_rho:.3f} g/cm³ outside expected range [0.80, 1.15]"
        )

    def test_mace_polar_medium_torch_to_jax_water_npt_f64(self, polar_checkpoint_f64, tmp_path):
        """NPT water simulation using MC barostat, float64.

        Verifies that the tight-minimization fix (force_tolerance=0.001) gives stable
        dynamics in float64 mode as well as float32.  Float64 params require HIGHEST
        TP precision to avoid float64→float32 downcast in TensorProduct accumulators.
        """
        from jax_md_cli import runner

        output_dir = tmp_path / "run_water_npt_mace_polar_mc_f64"
        md_config = _build_water_mc_barostat_config(
            str(polar_checkpoint_f64), "mace_polar", output_dir,
            steps=4000, dtype="float64"
        )
        print("Running NPT water simulation with MC barostat (mace_polar, float64)...")
        runner.run_simulation(md_config)

        thermo_csv = output_dir / "thermo.csv"
        assert thermo_csv.exists(), "thermo.csv not produced"
        mean_rho = _mean_density_last_half(thermo_csv)
        print(f"Mean water density (MACE-POLAR-1-M, MC barostat, float64): {mean_rho:.4f} g/cm³")
        assert 0.80 < mean_rho < 1.15, (
            f"MACE-POLAR-1-M water density {mean_rho:.3f} g/cm³ outside expected range [0.80, 1.15]"
        )