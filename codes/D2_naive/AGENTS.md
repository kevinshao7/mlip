# D2 Naive Fine-Tuning Instructions

`data/target_all.xyz` is the authoritative DFT dataset for this workflow. It
contains newly obtained nitrogen-bearing configurations; do not substitute an
older oxygen-only or residual/delta dataset unless the user explicitly asks.
The current `target_train.xyz`, `target_valid.xyz`, and `target_test.xyz`
splits are derived from that full data source.

Use `launch_single_gpu.py` or `trainmace.py`, never `run_train.py` directly.
Each real launch creates a unique datetime-named run folder containing
models, checkpoints, and logs. Do not overwrite or reuse a prior folder.

`hyperparameters.txt` at this workflow's root is the single, continuous living
experiment record. Every launch appends its resolved hyperparameters, command,
run path, training-log path, checkpoint-model path, and compiled-model path.
Append a dated note each time you analyze a run or decide on a follow-up.
Include the exact evidence
(metrics and log/plot location), diagnosis or uncertainty, proposed next test,
why that test is informative, and the one factor it will change. Keep the
original launch settings intact. Before launching a child run, append its plan
to this single record and put the same one factor in the child's
`--changed-parameter` / `--change-note` fields.

**Update code and experiment logs together.** In the same task that changes
training code, defaults, data, or this README, append a dated note explaining
that change and its rationale to this workflow's `hyperparameters.txt`.
Likewise, when a diagnosis changes the standard procedure, update the code or
README in that same task. Do not leave one side undocumented. Users may run
`python launch_single_gpu.py` with no extra arguments for the current baseline;
a unique run folder is created and the single experiment log is appended.

When a next experiment is approved, make it the active default in
`trainmace.py` and record the same setting in `hyperparameters.txt` in the
same task. The user must be able to run the exact prepared test with only
`python launch_single_gpu.py`; never require extra CLI flags for a test that
has already been prepared.

For every iteration:

1. Reproduce or select a documented baseline.
2. Change **one thing only**.
3. Supply `--changed-parameter` and `--change-note`.
4. Append the diagnosis and proposed next test to the parent's
   `hyperparameters.txt`.
5. Evaluate validation metrics against the immediate parent run before making
   the next change.

Changing data, split, architecture, foundation checkpoint, optimizer, learning
rate, loss weights, precision, or scheduling simultaneously invalidates a
causal comparison. Preserve unsuccessful runs and their hyperparameter files.
