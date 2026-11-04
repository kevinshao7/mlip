# Isolated checks

From the workflow directory, run:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Use `-p 'test_p10_batch.py'` (or another module) for a focused check. Tests use
temporary directories, fake sbatch and tiny CPU fixtures. They do not submit
Slurm jobs or run production MD, training or ORCA. No retained Python script
was classified as a disposable one-off test; the four test modules cover
production invariants and remain here.
