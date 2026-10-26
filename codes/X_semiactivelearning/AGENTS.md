# X semi-active-learning campaign rules

This is a new, self-contained pressure campaign.  It starts from foundation
`polar-1-s`; only an approved campaign fine-tune may become the parent of a
later campaign MD run.

Run no stage by directly calling a legacy B/C/D script.  Use `campaign.py` so
that every action is recorded in `outputsfull/X_semiactivelearning/`.

The pressure ladder is 10, 17.7828, 31.6228, 56.2341, and 100 GPa at NH3/H2O
R1.  Temperatures and target densities are interpolated in log10(pressure)
from the existing Uranus profile.  There is one MD seed per condition.

Species analysis uses nearest-heavy-atom H assignment, exactly as the existing
B production analysis.  The only base species are H2O, HO, H3O, NH3, NH2, and
NH4. The inherited formatter spells the ammonia family as H3N, H2N, and H4N;
these are the same allowed species. Any other complete component is a candidate event. A cluster is the
event component is diagnosis-only. It anchors selection but is never included in
DFT/training data: each DFT cluster contains the closest allowed component as
its centre plus that centre's five nearest allowed complete molecular neighbours
under PBC (six molecules total).
Generate diagnostic/acquisition data only when non-base species exist and their complete-component
fraction is below 5%; otherwise flag the condition for human review.

Before selecting DFT clusters, inspect the automatically generated
`analysis/pressure_volume_time_series.png`. It plots raw virial pressure and
cell volume against production time; a visible volume drift or sustained offset
from the target pressure means the frames are not an equilibrium training set.

Never automatically decide that a model needs fine tuning.  A diagnostic set
of at most 10 event-centred clusters is DFT-labelled and reviewed through its
parity plot and species plot.  An energy error around 50 meV/atom is a guide
for human review, not an automatic threshold.  Only `campaign.py approve`
may authorize the 50-cluster acquisition set.  Record the scientific reason
in that approval.

Always collect and evaluate parity from every successfully validated DFT frame,
including a set smaller than the 30-frame fine-tuning minimum. The minimum
only gates training; it never suppresses validation diagnostics.

Fine-tuning training data must come only from validated campaign DFT outputs
under `outputsfull/X_semiactivelearning/cycles/*/dft/acquisition/`. Do not
merge a legacy D2 or other external DFT training set; Polar-1-s is the model
initialization, while fixed validation/test files are evaluation-only.

Autonomous fine tuning uses CUDA device 1 (the second GPU) in 20-epoch
continuation batches. After each batch, parity is regenerated against the
latest model before any further batch may start; newly collected DFT data is
therefore incorporated at that boundary. Preserve every batch in model lineage
and training history.

The fixed D2 validation/test sets are never changed.  Validated newly acquired
DFT frames are appended to a cumulative campaign training pool.  Fine tuning
must name precisely one changed factor and preserve parent model/data hashes.

`provenance.jsonl` is append-only; do not edit it.  `campaign_summary.csv` and
each cycle's `manifest.json` are derived, human-readable indexes.  Preserve
raw MD, cluster, ORCA input/output, Slurm, and training artifacts.

On `dungeon1`, use `campaign.py dft ... --run-local` after preparation: it
runs one ORCA job using 20 of 24 CPU ranks through the environment's `orca_qc`
launcher after loading `mpi/openmpi-x86_64`, and skips normally terminated
outputs on restart. Do not launch multiple local ORCA jobs concurrently.
`--submit` is for a host with Slurm's `sbatch` available. Prepared inputs may
only be replaced with `--replace-prepared`, which archives them (including a
failed `.partial` output) and refuses if any ORCA output terminated normally.
Never change SCF parameters or retry a failed DFT calculation automatically.
The local runner preserves its `.out.partial`, classifies it, records it in
`dft_failure_report.json`, and continues with the next cluster. Only normal
ORCA terminations can be collected into training data.

For no-agent autonomy, use `autonomous.py` and change only its versioned
`autonomous_policy.json`. Do not add an LLM or agent decision point. GPU 0 is
the next-pressure MD worker; GPU 1 is reserved for parity/fine tuning. Never
idle GPU 0 waiting for DFT: immediately before each MD launch, read the latest
completed model from model lineage and record it in the MD manifest.
