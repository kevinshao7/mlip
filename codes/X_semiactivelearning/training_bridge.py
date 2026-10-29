"""Campaign adapter for the existing validated fine-tuning launcher recipe."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MLIP = ROOT.parents[1]


def main():
    # Retain the shared training recipe and registry, with campaign-owned logs.
    argv = sys.argv[1:]
    gpu_index = argv.index("--gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = argv[gpu_index + 1]
    forwarded = argv[argv.index("--") + 1:]
    path = ROOT.parent / "D2_naive/trainmace.py"
    spec = importlib.util.spec_from_file_location("campaign_training_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    for name in ("append_experiment_log", "append_run_outcome"):
        original = getattr(recipe, name)

        def mirrored(*args, _original=original, **kwargs):
            source = path.parent / "hyperparameters.txt"
            offset = source.stat().st_size if source.exists() else 0
            _original(*args, **kwargs)
            with source.open("rb") as handle:
                handle.seek(offset)
                entry = handle.read().decode("utf-8")
            with (ROOT / "hyperparameters.txt").open("a", encoding="utf-8") as handle:
                handle.write(entry)

        setattr(recipe, name, mirrored)
    sys.argv = [str(path), *forwarded, "--runs-root", str(
        MLIP / "outputsfull/X_semiactivelearning/acquisition_p10/runs")]
    recipe.main()


if __name__ == "__main__":
    main()
