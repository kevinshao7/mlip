"""Campaign adapter for the existing validated fine-tuning launcher recipe."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLIP = ROOT.parents[1]


def main():
    import campaign
    # Retain the shared training recipe and registry, with campaign-owned logs.
    argv = sys.argv[1:]
    gpu_index = argv.index("--gpu")
    if os.environ.get("SLURM_JOB_ID"):
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Missing Slurm-assigned GPU visibility")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = argv[gpu_index + 1]
    forwarded = argv[argv.index("--") + 1:]
    path = ROOT.parent / "D2_naive/trainmace.py"
    spec = importlib.util.spec_from_file_location("campaign_training_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    # Logging resolves through SCRIPT_DIR; constants for data/model dependencies
    # were already initialized from the shared recipe. Keep one X lab notebook.
    recipe.SCRIPT_DIR = ROOT
    sys.argv = [str(path), *forwarded, "--runs-root", str(
        campaign.OUT / "runs")]
    recipe.main()


if __name__ == "__main__":
    main()
