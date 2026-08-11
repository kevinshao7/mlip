#!/bin/bash
#SBATCH --partition=standard --time 12:00:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH -o unforcedslumlogvel_v1.0e+05_c0.txt
module purge
module load orca/6.1.1
