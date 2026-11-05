# MD tasks

Use `python md/p10_batch.py --help` from the workflow directory for the named
P10 MD plan, checking and restart entry point. The command preserves the
existing top-level adapter and shared `production/` implementation. Scientific
settings and source hashes stay frozen; this CPU DFT task does not prepare or
submit MD. Generic pressure progression remains `autonomous.py`.
