# DAIS Python and GitHub MACE

This record describes DAIS only. For the later Viper installation of latest
GitHub development MACE, see [VIPER_ENVIRONMENT.md](VIPER_ENVIRONMENT.md).

Verified 2026-11-04 on dais11.mpcdf.mpg.de. Use this existing environment:

```bash
source /dais/fs/scratch/kshao/mlipenv/bin/activate
python -c 'import sys,mace; print(sys.executable); print(mace.__file__)'
```

For DAIS GPU Slurm jobs, the existing wrappers load `gcc/14` and `cuda/12.8`.
The explicit interpreter is `/dais/fs/scratch/kshao/mlipenv/bin/python` (Python
3.11.15). System `/usr/bin/python3` is 3.6; the historical Windows path does
not apply here. Set `XDG_CACHE_HOME` to the repository's `outputsfull/.cache`.

MACE is installed editable from this repository's `mace/` GitHub checkout:
[v0.3.17](https://github.com/ACEsuit/mace/releases/tag/v0.3.17), commit
`e37c8583a8b1b586e5888fa6dca86e8f591e2606`. This was the latest published
release when checked. Develop's unreleased 0.3.18 was not selected. Imports
must resolve to `mlip/mace/mace/__init__.py`; keep that source checkout present.
The parent repository records MACE as a git submodule; initialize it after cloning.

The required Polar dependency graph_longrange was built from GitHub
WillBaldwin0/graph_electrostatics tag v0.4.4, commit
`6a86de5e3ed35fd86a55fc046aa085fe48a72764`, replacing 0.4.0.
PyTorch 2.8.0+cu128, ASE 3.26.0 and NumPy 2.0.2 were preserved.

No pip commands were used. `tools/build_mace_source.py` uses setuptools PEP 517/660
and PyPA installer. Standalone uv supplied isolated build tools (setuptools
79.0.1, wheel, installer); dependency requirements were checked against the
existing environment before installation. Use explicit target/source/output
arguments and a fresh output directory; default operation builds only, while
`--install` publishes packages with rollback copies. Do not automatically
upgrade scientific dependencies or invoke this helper for routine debugging.

Build evidence is under
`outputsfull/environment_builds/20261104T004139655904Z/`:
`before.json`, `source_install.log`, `cpu_smoke.log`, and
`editable_source/{build_manifest.json,install_record.json,wheels,rollback}`.
The install record maps original paths to backups. The shared old tests/scripts
namespace was audited and mdinterface-owned files were restored; the helper now
backs up those namespaces per file. Preserve rollback data for recovery.

Validation: source import/version passed; the existing foundation Polar 1 L
checkpoint produced finite isolated neutral H2O energy and forces on CPU
(-2079.865478515625 eV; forces shape 3 x 3). This is a compatibility check,
not a scientific accuracy benchmark. CUDA/GPU execution was not tested.
No MD, training or ORCA jobs were submitted. Existing download-time provenance
records retaining MACE 0.3.16 describe the original model download; they do not
identify the newly installed runtime. Foundation checkpoint and scientific
simulation settings were unchanged.
