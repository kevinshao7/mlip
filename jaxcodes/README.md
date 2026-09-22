# JAX-MD CLI workflows

> [!IMPORTANT]
> **Use the `dev` branch of the external `jax-md-cli` package.** The `dev`
> branch should now work for MACE-POLAR and is required by the configurations
> in this directory that set `model_type = "mace_polar"`.

These are project-side configurations and launchers; `jax-md-cli` is an
external Python dependency. Before submitting or running a MACE-POLAR job:

```bash
cd /path/to/jaxmd-cli
git switch dev
git pull
git branch --show-current   # must print: dev
```

The MACE-POLAR Slurm launchers and Python potential loader intentionally abort
with a clear error if the external checkout is missing, detached, or on any
branch other than `dev`.
