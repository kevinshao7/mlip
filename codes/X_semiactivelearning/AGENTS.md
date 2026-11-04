# X workflow: agent entry point

## First read: current task and source of truth

Keep work within the user's requested stage. Current scope is **prepare code
only**; no `sbatch`, production MD, ORCA, or training until explicitly requested.
User-authorized read-only Slurm queries, model preparation and isolated CPU tests
are distinct from production launches. Do not ask again for already authorized
reversible preparation. Report actual blockers plainly.

1. Read `p10_batch_active.json` and its `plan.json` for the prepared P10 batch.
2. Read `P10_BATCH.md` for its physical/resource settings and prerequisites.
3. Open the relevant function/test from the routing table below.
4. Use `DEBUGGING.md` only for the symptom you are diagnosing; the append-only
   `hyperparameters.txt` holds historical evidence, not active defaults.

The active P10 deployment is ten **foundation polar-1-l** replicas, 10 GPa,
1920 K, R1, 1 ns production each plus 2+2 ps ramps, 0.5 fs and 2000 frames.
Each Slurm job requests one H200/12 CPUs/64G/24 h; each replica has its own
six-job continuation chain. No GPU array, shared-node/co-start requirement,
or dependencies on other replicas. Six segments are an uncalibrated estimate,
not a guarantee that 1 ns fits; DAIS gpu1's observed maximum is 24 h per job.
The foundation checkpoint is already cached and frozen; use its portable plan
hash. The old request for a fine-tuned checkpoint is superseded for this batch.

## Which code owns the change?

| Task | Edit/read first | Narrow useful check |
| --- | --- | --- |
| P10 duration/resources/seeds/submission topology | `p10_batch.py:prepare`, `write_launchers` | `test_p10_batch.py` |
| Generic MD code generation | `campaign.py:command_md` | preparation/AST test in `test_p10_batch.py` |
| Restart state, progress, production integrator | `md_segments.py` | `test_md_segments.py` |
| Whole-molecule PBC and geometry checks | `cluster_quality.py`, `campaign.py:build_cluster_worker` | `test_cluster_quality.py` |
| Sampling partitions/window/pressure gate | `workflow_policy.py`, `campaign.py:command_clusters` | `test_cluster_quality.py`, relevant `test_active_workflow.py` cases |
| ORCA input/output charge, spin, method, gradients | `orca_output_validation.py`, `campaign.py:dft_input` | `DFTValidationTests` in `test_active_workflow.py` |
| Direct-method pilot | `method_pilot.py`, `require_method_verified` | existing pilot-package smoke test; target ORCA needs a real pilot |
| Viper/BlueHive environment/resources | `site_profiles.json`, `site_dft.py`, P10 generator | shell syntax/site-package test |
| Fine tuning and parent promotion | `launch_single_gpu.py`, `training_bridge.py`, `campaign.py:command_finetune` | pool/lineage tests; optimization only if authorized |
| Generic pressure progression | `autonomous.py`, `autonomous_policy.json` | `ControllerTests` in `test_active_workflow.py` |

P10 uses `p10_batch.py` and its named output root. `autonomous.py` uses the
separate generic pressure campaign; do not run it to manage this P10 bundle.
B's MD template and D2's training recipe remain supporting dependencies; do not
move them without tracing callers. `deprecated/` is audit history, not an entry
point. No new work uses polar-1-s or the retired learning-rate tournament.

## Edit, verify, finish

Use the verified Python 3.11 mlipenv (root AGENTS.md); don't rediscover or install
an environment for each edit. Search source with scoped `rg`, and read narrow
function ranges. Avoid listing thousands of generated ORCA inputs or scanning
large trajectories when an existing hash-backed audit answers the question.

Run the relevant focused test once after the final code edit. Use the full
suite only for cross-cutting changes or unresolved failures. Code generation
is tested by writing temporary files and compiling them, not launching MD.
The submission test uses a fake `sbatch` and verifies all 60 prospective jobs'
per-replica dependencies; never replace it with a real submission smoke test.
For docs-only work, check links/paths and whitespace; do not rerun code tests.

`source_sha256` and config/model hashes intentionally freeze a launch plan.
When a hashed source/default changes, test the generator, prepare a fresh dated
bundle, update the active pointer and archive superseded *code* bundles. Preserve
scientific outputs. Never edit hashes to bypass a mismatch. `cache-foundation`
reuses the existing cache; a cached 123.8 MB model needs no new network download.
Docs-only edits do not invalidate source hashes and need no plan regeneration.

Append one dated entry to `hyperparameters.txt` with observation/evidence,
diagnosis/uncertainty, next test, why informative, and the changed factor.
Update the relevant README/default in the same task when procedure changes.
Do not repeat the entire notebook in each entry. Record no launch if none ran.
Commit/push only under explicit user authorization; don't include cached models,
trajectories, raw scratch or unrelated untracked outputs.

## Scientific invariants

- Molecular formal charge is H-3N-2O and must match ORCA input, computed output
  and extxyz exactly. Verify multiplicity separately (singlet/even electrons).
  Unsupported molecular elements fail. Neutral isolated E0 atoms are a distinct
  reference-state exception with explicit charge 0 and verified multiplicities.
- Active labels are dispersion-free wB97M/def2-TZVPD absolute energies in eV
  and forces in eV/Angstrom. Exclude V/D4rev total labels. A direct LibXC input
  is not certified merely because its syntax passes: require the one-geometry
  pilot against preserved C6 labels before production DFT. Do not invent a
  plain !wB97M alias or loosen tolerances automatically.
- Exactly 2000 P10 DFT structures: 1950 acquisition +50 evaluation. Six allowed
  complete molecules per cluster; one centre plus five nearest allowed molecular
  centroids. Unwrap each molecule internally, then translate its whole image.
  Reject changed internal distances (>1e-8 Angstrom), nonfinite/coincident
  geometry, wrong charge/spin or duplicate frame/centre keys. Report shortest
  contacts without inventing physical cutoffs.
- Retain existing B/X species assignment (nearest H/N/O including H, plus
  heavy-atom bonding). Allowed H2O, HO, H3O, H3N/H2N/H4N only. Zero illegal
  components is eligible; fraction >=5% is blocked. Illegal molecules never
  enter clusters. Uniform frame sampling is not event-biased.
- Review pressure/volume and species; a longer run does not prove equilibrium.
  Approval is tied to the analysis hash. Sampling must honour actual start/stride
  and evaluation blocks, including all centres from each excluded frame. Never
  train/validate on held-out evaluation frames; correlated frames aren't IID.
- Production keeps a persistent MTK integrator and checkpoints full extended
  state, with output offsets and version/model/config checks. Do not restart
  from coordinates alone, repeat ramps, rethermalize or reset chains every
  chunk. Preserve uncommitted tails before checkpoint rollback.
- Pressure advancement is +5 GPa only for current-model/current-data held-out
  force-component RMSE strictly <20 meV/Angstrom with >=30 valid evaluation
  clusters. Equality/NaN/missing/stale metrics fail. Repeated gate use is adaptive
  validation, not an untouched final test.
- Every fine-tune uses the workflow launcher, unique dated run and pre-optimization
  notebook entry. Preserve the documented recipe and change one training factor
  per child; compare parent/saved child on the same split before promotion.
  Record failed attempts and retain the parent if the child regresses.
- Preserve all raw ORCA/stdout/property/Slurm/checkpoint evidence. Failed DFT
  is skipped without SCF tweaks or retries. GPU visibility belongs to Slurm.

## Current validation limits (2026-11-04)

No production jobs were submitted in this preparation session. The direct-method
ORCA pilot and actual 2000-cluster set remain pending future MD/DFT work. All
266 historical C6 frames passed basic finite/charge/spin/PBC checks, but missing
parent MD prevents certifying their whole-molecule extraction. CPU restart tests
are not a polar-1-l H200 throughput or GPU execution benchmark. See DEBUGGING.md
for evidence locations and specific failure recovery.
