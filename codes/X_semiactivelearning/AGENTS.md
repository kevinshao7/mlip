# Active X campaign rules

The user-approved workflow is periodic polar-1-l MD, allowed-molecule cluster
extraction, dispersion-free wB97M/def2-TZVPD DFT, fine tuning, and held-out force
error characterization. Start at 10 GPa and increase pressure by exactly 5 GPa
only when current-pressure force-component RMSE is strictly below 20 meV/Angstrom.
At or above the threshold, acquire additional unique clusters and continue
fine tuning at the same pressure. This user direction supersedes the old
energy-triggered/log-pressure campaign; preserve its evidence in deprecated/.

Use campaign.py and autonomous.py, not legacy B/C/D entry points directly.
The shared D2 training recipe and B MD template are supporting dependencies;
do not move them without updating and checking their callers. No new work uses
polar-1-s. Default output namespace is outputsfull/X_semiactivelearning/wb97m_step5;
never merge historical V or D4rev total labels or reuse their model lineage.
Source/launch code belongs here, scientific outputs under outputsfull/.

Do not invent a plain ORCA !wB97M alias. The proposed direct LibXC route must
pass the single-geometry method pilot against preserved C6 subtraction labels
before production DFT. Reject any evaluated D3/D4/VV10 correction. Keep all raw
input/output/property/scratch artifacts. Formal molecular charge H-3N-2O must
match ORCA input, computed output, and extxyz exactly; multiplicity is separately
verified as singlet with even electron count. Unsupported molecular elements
fail. Neutral isolated atomic references have charge 0 and independently
verified ground-state multiplicities; this exception applies only to E0s.

Clusters contain one allowed complete molecule plus five nearest allowed
complete molecular neighbours, each wrapped about its own centre anchor under
PBC. Allowed species: H2O, HO, H3O, NH3/NH2/NH4 (H3N/H2N/H4N spellings).
Non-base components are never centres/neighbours. Require less than 5% non-base
components; zero non-base components is eligible. Uniform frame sampling is
independent of non-base events. Deduplicate by trajectory, frame, centre.

Review pressure_volume_time_series.png and species plots and record acquisition
approval with campaign.py approve before extraction. Known compression transients
must never be called equilibrium samples. Do not silently change temperature,
pressure ramp, timestep, model, barostat, or ensemble to fix equilibration.

Reserve temporal frame blocks for evaluation; never use their labels for
training or validation, even through a different molecular centre. Validation
assignments stay fixed as acquisitions grow. Nearby blocks may remain correlated;
this is not independent-trajectory generalization. The repeatedly consulted
pressure gate is an adaptive validation set, not an untouched final test.
Require at least 30 valid training-pool structures and 30 held-out evaluation
clusters. Report both energy and force errors; only held-out force RMSE gates
pressure. NaN, unavailable metrics, wrong model/data hashes, and equality at
20 meV/Angstrom never pass. Failed ORCA jobs are preserved and skipped without
SCF changes/retries; interrupted partial files require inspection.

Use launch_single_gpu.py for every batch. No flags are required for the active
prepared batch. Preserve parent, lr=1e-4, 20 epochs, no SWA and existing loss
recipe while changing acquisition_dataset alone. New unique timestamped runs,
pre-optimization records and outcomes go to this workflow's hyperparameters.txt.
Append evidence, diagnosis/uncertainty, proposed next test, why informative,
and the one changed factor for every inspection/change/decision. Record parent
metrics on the same frozen snapshot before comparing the child. Never silently
promote a regressed saved model.

ORCA runs on Viper or BlueHive through separate site profiles/Slurm packages.
GPU MD/training runs directly on dungeon or through DAIS GPU Slurm wrappers.
Retain Slurm-assigned CUDA_VISIBLE_DEVICES. Paths and modules are centralized
in site_profiles.json and Slurm templates. Log files belong under outputsfull/.
No concurrent ORCA retries and no agent-controlled optimizer tournaments in the
active workflow. Historical controllers/tournament scripts are reference-only.

Do not commit/push unless explicitly requested by the user. Preserve generated
scientific data and prior run directories. Validate code with py_compile and
isolated workflow tests before finishing; report runtime/dependency blockers.
