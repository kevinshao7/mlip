# Current data protocol — 2026-11-03

**Active campaign:** [X_semiactivelearning](codes/X_semiactivelearning/README.md).
The current default is a new `wb97m_step5` namespace: polar-1-l MD, PBC-aware
clusters, direct dispersion-free wB97M DFT after method-pilot verification,
training and held-out force-component RMSE. Stay at each pressure until RMSE
is strictly below 20 meV/Angstrom, then increase by 5 GPa (10 through 100 GPa).
C6 is supporting label/provenance evidence. The inventory below describes the
previous C6 release and historical P10 data, not the active launcher.


The currently prepared P10 deployment is [the foundation batch](codes/X_semiactivelearning/P10_BATCH.md):
ten one-GPU polar-1-l replicas, 1 ns production plus 4 ps ramps via resumable
24 h jobs (six allocations per replica, ten independent submission chains),
2,000 saved frames each. Its 2,000-label Viper budget includes 50 held-out
clusters. Whole-molecule PBC geometry and reviewed-window checks precede DFT.
The foundation checkpoint is pinned; no jobs have been submitted.

## Active scientific direction

Use dispersion-free wB97M/def2-TZVPD energy AND force targets. Do not mix
wB97M-V or wB97M-D4rev total labels into training, validation, or test sets.
The current C6 release obtains wB97M-only targets by subtracting the printed
D4rev energy and gradient from existing ORCA calculations. D4rev in raw
provenance is therefore expected; it is not the target method. A new direct
no-dispersion ORCA production route is not yet implemented by C6, whose parser
specifically validates the historical D4rev source method.

## Workflow and supporting tools

| Role | Current location | Status |
| --- | --- | --- |
| Target conversion and validation | `codes/C6_wb97m_nodisp/prepare_data.py` | Active data preparation; unique timestamped outputs |
| Conversion checks | `codes/C6_wb97m_nodisp/test_prepare_data.py` | Energy/force reconstruction and rejection tests |
| Released labels and audit | `outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/` | 266 accepted of 363 outputs; 97 excluded without normal termination |
| Raw evidence | `outputsfull/acquisition/outputs/` | Preserve original stdout, including failures |
| MD, species, cluster selection and DFT orchestration | `codes/X_semiactivelearning/campaign.py` | Supporting campaign workflow; read its AGENTS.md |
| Existing training launcher | `codes/X_semiactivelearning/launch_single_gpu.py` | Active no-dispersion launcher; prior version archived in deprecated/ |
| Atomic reference evidence | `codes/C5_atomizationD4rev/` | Verify isolated-atom dispersion vanishes before reuse for wB97M |
| Optional dispersion addition | `codes/C6_wb97m_nodisp/d4_inference.py` | Separate inference option; backend parity remains unverified |

The current plan avoids dispersion-inclusive training. Adding D4 at inference
is a separate scientific decision, not a requirement to train wB97M-only.

## Label and experiment invariants

C6 writes `REF_energy = E_total - E_D4` in eV and
`REF_forces = -(gradient_total - gradient_D4)` converted to eV/Angstrom.
These are absolute electronic energies, without atomization subtraction.
Clusters are isolated with PBC=False; parent MD is periodic. Require normal
ORCA termination, final energy, finite gradients, atom-order agreement,
formal charge H - 3N - 2O matching input/output/extxyz, and independently
verified multiplicity. Preserve source identities and SHA256 provenance.

For a controlled target-dispersion comparison, preserve the parent split by
source identity (the historical P10 package has 213 train / 53 validation),
and remove dispersion from every evaluation target too. Do not independently
resplit the 266 C6 frames. Correlated MD structures are not independent samples.
A no-dispersion child must use the workflow launcher, a unique dated directory,
and a pre-optimization experiment record. No no-dispersion child or validation
metrics are recorded in the inspected release. The old D4rev launcher is archived. The active launcher now requires the new
wB97M campaign pool and audited E0s; it does not import this release automatically.

## 10 GPa trajectory availability

The latest named campaign `p10_2gpu_6h` documents two completed 2000-frame
polar-1-l replicas, nominally 10 GPa, 1920 K, NH3/H2O R1, 0.5 fs timestep,
and 45 ps production holds. Its README/log record mean virial pressure
3.6717 GPa and volume contraction about 4148 to 2280 Angstrom^3; final
200-frame means remain about 5.85/5.88 GPa. These are compression transients,
not established equilibrium 10 GPa data. The prior user-approved extraction
of 1000 clusters retained that caveat. These campaign MD xyz files were not
found in this local checkout; recover them from the originating machine before
re-extraction. Available ORCA inputs and labels do not replace a trajectory.

There IS a local legacy trajectory at:

`outputsfull/conditionsproduction/P10GPa_R1/production_seed_941745243_P_10GPa_T_1920K_R1_initial_density_0.2_profile_density_0.344.xyz`

A full streaming scan verified 223,477 complete frames, each with 350 atoms
(16,320,160,968 bytes), with periodic cell metadata. The associated generated job
`codes/B_conditionsproduction/expand/production_P10GPa_R1.py` specifies
IsotropicMTKNPT, 10 GPa, 1920 K, 0.5 fs steps and polar-1-s. This is legacy
sampling, not the new large-model campaign; the script alone does not certify
run completion or equilibrium. Other local P10 compositions and stride100
copies also exist. Before using these as a new cluster pool, verify actual
frame completeness, recorded model/settings, stationary pressure/volume and
species eligibility. Do not merge legacy labels into campaign training.

For campaign extraction, use `campaign.py`: uniformly sample eligible frames,
select one allowed complete molecule plus five nearest allowed complete
molecular neighbours, wrap each cluster about its own centre under PBC, and
deduplicate by (trajectory_id, frame, center_component). Review pressure/volume
and species diagnostics before acquisition. No new MD, DFT, extraction or
training was launched during this inspection.
