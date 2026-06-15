## Wrapper for mace.cli.run_train.main ##

from mace.cli.run_train import main

if __name__ == "__main__":
    main()

"""
python ./trainmace.py \
  --name="polar_ft_1m" \
  --model="PolarMACE" \
  --foundation_model="polar-1-s" \
  --train_file="./ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz" \
  --valid_fraction=0.05 \
  --energy_key="TotEnergy" \
  --forces_key="force" \
  --compute_forces True \
  --E0s "estimated"\
  --loss="weighted" \
  --stress_weight=0.0 \
  --force_mh_ft_lr=True \
  --default_dtype="float32" \
  --device="cuda"\
  --batch_size=1 \
  --max_num_epochs=20\
  --multiheads_finetuning=True\
  
plot predictons

python mace/scripts/eval_configs.py \
  --configs="ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz" \
  --model="polar_ft_1m_e15.model" \
  --output=predicted.xyz \



  """