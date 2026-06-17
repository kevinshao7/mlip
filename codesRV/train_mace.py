## Wrapper for mace.cli.run_train.main ##

from mace.cli.run_train import main

if __name__ == "__main__":
    main()

"""
sbatch submit_mace_polar_raven.slurm
tail -f gpu_debug.<jobid>.out
"""