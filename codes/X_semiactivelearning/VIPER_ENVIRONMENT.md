# Viper Python and GitHub MACE

Verified 2026-11-04 on viper02. Latest upstream MACE requires Python >=3.11;
Viper's existing mlip_clean is Python 3.10, has no Torch/MACE installation,
and was preserved. The DAIS environment path is unavailable on this host.

Use the new Viper environment:

```bash
source /viper/ptmp1/kshao/mlip/.venv/bin/activate
python -c 'import sys,mace; print(sys.executable); print(mace.__version__); print(mace.__file__)'
```

Explicit interpreter: `/viper/ptmp1/kshao/mlip/.venv/bin/python`, Python 3.11.11.
MACE 0.3.18 is installed editable from the official ACEsuit/mace GitHub checkout
at `mlip/mace/`, upstream develop commit
`7215fadd3b591444ab2c07634e84419685a48e87` (2026-10-28), matching upstream HEAD
when checked. This is the latest development source, not a published-release
claim. Imports resolve to `mlip/mace/mace/__init__.py`. Keep the checkout in place.
The submodule working checkout was advanced from the parent repository's old
v0.3.17 pin; no parent commit or push was made.

Standalone uv 0.13.7 built the editable checkout using setuptools; no pip
installation command was used and no registry MACE distribution was selected.
Other runtime dependencies were resolved by uv. The Polar dependency
`graph_longrange` 0.4.4 was built from GitHub WillBaldwin0/graph_electrostatics
tag v0.4.4, commit `6a86de5e3ed35fd86a55fc046aa085fe48a72764`.
Torch 2.8.0, NumPy 2.0.2 and ASE 3.26.0 match the documented DAIS versions;
Torch includes CUDA 12.8 support. This node reported CUDA unavailable, so GPU
execution was not validated. MACECalculator/PolarMACE imports, finite CPU tensor
operations and `python -m mace.cli.run_train --help` passed; no training, MD,
ORCA or foundation model download was performed.

Build project, exact dependency lock, installation logs and source provenance:
`outputsfull/environment_builds/viper_mace_20261104/`, especially
`verified_install.json`, `pyproject.toml`, `uv.lock`, `source_install.log`,
`graph_source_install.log` and `train_help.log`.
MACE's installed direct_url.json identifies the editable local GitHub checkout;
the Polar dependency's direct_url.json records its GitHub URL and exact commit.
Caches remain under `outputsfull/.cache`. Set `XDG_CACHE_HOME` there for model use.

The old mlip_clean environment is unchanged and remains usable for the earlier
DFT preparation checks. To use the new environment in a Viper DFT helper,
activate it or set PYTHON_COMMAND to the explicit interpreter above. DAIS
wrappers retain their site-specific environment. CPU import checks do not
establish GPU performance or scientific prediction accuracy.
