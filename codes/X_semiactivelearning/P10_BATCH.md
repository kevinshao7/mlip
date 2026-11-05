> **2026-11-05 current CPU DFT preparation:** the published 2000-cluster release
> now exists. Use [dft/README.md](dft/README.md), `cpu_dft_active.json` and
> `bash submitcpu.sh` for 200 separate Viper jobs with 20 CPUs and 10 clusters
> each. The old 120-hour array description below is historical and exceeds
> Viper's current 24-hour partition limit. No CPU jobs were submitted during
> this preparation. Earlier statements about absent geometries describe local
> November 4 availability; preserve the original MD source-plan provenance.

> **2026-11-04 selected and implemented path:** calculate
> `wB97M-D4rev def2-TZVPD EnGrad RIJCOSX` and subtract both printed D4 energy
> and forces for electronic-only labels. Generation, collection and provenance
> implement this route in the fresh plan selected by `p10_batch_active.json`.
> The old direct-route bundle is superseded and preserved. Submission preflight
> now stops at missing `dft_package.json`; no production jobs were submitted.
> See [DFT provenance](DFT_PROVENANCE.md).

# Prepared foundation polar-1-l P10 ensemble and Viper DFT batch

No jobs have been submitted and no MD, ORCA, or training has run. The current
plan is linked by `p10_batch_active.json`; older dated plans are retained for
audit and are superseded. The active plan uses **foundation polar-1-l** per the
user's 2026-11-04 direction. The earlier fine-tuned-checkpoint requirement is
superseded for these ten runs.

## MD model, 1 ns target and 24-hour segments

Ten independently seeded replicas are prepared as **ten separate Slurm jobs**, each with
**one H200, 12 CPU cores, 64 GiB host memory and a 24-hour limit per allocation**
on DAIS. This is 240 GPU-hours of requested wall-clock allocation per set of ten independent jobs if all ten
are later submitted. Slurm owns GPU visibility. Each replica has a distinct
output directory and seeds NumPy, Python and Torch before packing/velocities.
The working gcc/14, cuda/12.8, mlipenv setup is retained.

Foundation polar-1-l was downloaded once into `outputsfull/.cache`, copied to
the active output namespace and pinned by SHA256:

`9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b`

All ten jobs check the same frozen bytes. The portable plan records the release
URL/hash; on another GPU host, `python p10_batch.py cache-foundation` prepares
those same bytes and rejects a different download. Model weights stay out of
Git. No fine-tuned model is selected by this plan.

The target is **1 ns (1,000 ps) production per replica**, plus unchanged
2 ps pressure and 2 ps temperature ramps: 2,008,000 NPT steps at 0.5 fs, plus
100 initial NVT steps. Each replica saves 2,000 frames, every 1,000 steps
(0.5 ps). The earlier 176 ps bundle is archived.

DAIS `sinfo -p gpu1` reports a maximum allocation of **1 day**. Each invocation
of `submit_md_dais.sh` submits exactly **ten independent jobs**, one per replica,
with no arrays or dependencies. Each job requests 24 h and computes for at most
22 h before checkpointing. The 1 ns target remains unchanged; the historical
proxy estimates approximately 123 h per replica, so a single allocation may
not finish it. Further restart submissions require a separate launch decision
after reviewing progress. No continuation jobs are automatically queued.

`md_segments.py` saves positions, momenta, cell, reference-cell/volume variables,
thermostat and barostat chain coordinates/momenta, progress and output offsets
in atomic, pickle-free checkpoints. ASE version, Torch version, model/config
hashes and physical parameters must match on restart. Uncommitted output tails
are archived before replay, preventing duplicate frames. Initial thermalization
and ramps are not repeated after resuming.

The inherited runner recreated its NPT integrator every 20 steps during
production. This reset its chain state. The new production runner keeps one
persistent integrator and restores its complete state between allocations;
nominal temperature, pressure and damping settings remain the same. The existing
changing-target ramp schedule is retained. Tiny CPU restart fixtures matched
uninterrupted integration and preserved frame counts; no campaign MD was run.

Scientific settings remain 10 GPa, 1920 K, NH3/H2O R1 (50+50 molecules,
350 atoms), PBC=True, IsotropicMTKNPT, total charge 0, singlet, float32,
thermostat/barostat damping 100/1000 fs. Initial density stays 0.2 g/cm3;
profile density 0.344 g/cm3 is metadata, not an imposed volume. The checkpoint's learned-neighbour cutoff is 6.0 Angstrom (verified by CPU
checkpoint inspection); calculator long-range settings are retained. Each
actual run records its neighbour cutoff in `model_runtime.json`.

## Cluster quality and the 2,000-job budget

Assuming extraction from the ten new trajectories, the budget is exactly
**1,950 acquisition + 50 held-out evaluation = 2,000 ORCA clusters**, or 195+5
per equal-length replica. Ten seeds do not make adjacent frames independent.
The existing temporal frame partitions are retained, and evaluation frames
cannot enter training through a different centre molecule.

The code audit found and corrected atomwise minimum-image wrapping that could
tear a neighbour molecule at the half-cell boundary. Extraction now unwraps
each complete molecule internally, then translates the entire molecule to its
nearest centroid image relative to the selected centre. It selects one allowed
centre plus five nearest allowed complete molecules. Regression checks verify
PBC bond preservation and the nearest-five ordering.

Before preparation, `cluster_quality.json` must certify all 2,000 structures:

- six complete allowed molecules and preserved intramolecular distances
  (maximum numerical error <=1e-8 Angstrom);
- finite coordinates, isolated PBC=False/no cell, no coincident atoms;
- recomputed H-3N-2O charge, explicit singlet and matching molecule membership;
- unique trajectory/frame/centre sources across acquisition/evaluation;
- correct data partitions and membership within the reviewed analysis window.

Observed shortest contacts are reported; no new physical distance cutoff is
silently imposed. The established B/X species assignment (nearest H/N/O atom,
including H, plus heavy-atom bonding) is retained. Species labels remain H2O,
HO, H3O, H3N/H2N/H4N; illegal components never enter clusters.

The historical C6 pilot reference was inspected on CPU: all 266 frames pass
finite geometry/labels, formal charge, singlet/electron-count and isolated-PBC
checks. Its atom count ranges 17–24 and minimum pair distance is 0.815527
Angstrom. This does **not** certify historical whole-molecule preservation,
because original parent MD is absent locally. It is also not a quality report
for the future 2,000 geometries, which do not exist yet.

Known short P10 runs were compression transients. The new longer duration
alone does not prove stationary 10 GPa sampling. Inspect pressure/volume and
species, choose a stable tail if needed, and record the review before extraction:

```bash
python p10_batch.py analyze
# After inspecting the diagnostic plots, optionally restrict the window:
python p10_batch.py analyze --start-frame N
python p10_batch.py approve --reason "Record pressure/volume evidence and accepted window"
python p10_batch.py clusters
```

Analysis first checks all ten completion markers and all 2,000 complete raw
frames per replica. Sampling then honours the actual start frame and stride;
approval is tied to the analysis hash. Changing the window requires a fresh
review/approval, with the earlier record preserved.

## Viper resources and method

Viper array `0-199%20` comprises 200 groups of ten serial calculations, at most
20 groups concurrently. Each group reserves 20 CPU cores, 64 GiB and 120 h;
ORCA inputs retain `%pal nprocs 20`. Site paths, ORCA 6.1.1 module and MPI
layout come from the Viper profile. The array runner requires Python >=3.10;
`PYTHON_COMMAND` selects that interpreter. Its hash/charge preflight uses only
the standard library; NumPy/ASE are required for preparation/collection.

Labels are dispersion-free wB97M/def2-TZVPD EnGrad RIJCOSX, isolated clusters,
absolute energies in eV and forces in eV/Angstrom. ORCA evaluates D4rev; collection removes its printed energy and gradient,
checks reconstruction, and retains total/D4 audit labels. VV10 labels and
unsubtracted D4rev total labels are excluded. The direct LibXC equivalence
pilot is no longer required for this selected C6 subtraction route. Failed or partial ORCA jobs, stdout, exit
codes, scratch and property files are preserved without automatic SCF retries.

Prepared files include `md_replica_00_dais.slurm` through `md_replica_09_dais.slurm`, `dft_2000_viper.slurm` and
separate submission helpers. The helpers perform preflight and create Slurm
log directories before sbatch; none has been invoked. To inspect readiness:

```bash
python p10_batch.py check-md
```

After the future reviewed extraction, these prepare/check
commands do not submit anything:

```bash
python p10_batch.py prepare-dft
python p10_batch.py check-dft
```

Transfer code, the active dated plan/config, ORCA inputs/manifests,
the D4rev subtraction provenance, geometry audit and package index to the same relative
locations on Viper. Model weights/MD trajectories are unnecessary for already
prepared ORCA jobs. Transfer completed output stages back before collection:

```bash
python p10_batch.py collect
```

The generic X per-site packages are retained for audit. Use one submission route
per input; do not submit both the generic packages and the batch array. Scientific
outputs stay under `outputsfull/X_semiactivelearning/<run_id>/`. The single lab
notebook is `codes/X_semiactivelearning/hyperparameters.txt`.
