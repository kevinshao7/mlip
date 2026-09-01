#!/usr/bin/env python
"""Compatibility launcher for the Python 3 ORCA input generator.
  for f in expand/C3_DFTprod_stopH2_group_*.slurm; do echo "Submitting $f"; sbatch "$f"; done
BlueHive's ``python`` may be Python 2.  Keep this file parseable there so
``python expand_dft_jobs.py`` can re-exec the real Python 3 implementation.
"""
from __future__ import print_function

import os
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    implementation = os.path.join(script_dir, "_expand_dft_jobs_py3.py")

    if sys.version_info[0] >= 3:
        sys.path.insert(0, script_dir)
        import mlip.codes.C3_DFTproductionstopH2.expand_dft_jobs_py3 as expand_dft_jobs_py3

        expand_dft_jobs_py3.main()
        return

    candidates = []
    env_python = os.environ.get("PYTHON3")
    if env_python:
        candidates.append(env_python)
    candidates.extend(["python3", "python3.12", "python3.11", "python3.10"])

    for executable in candidates:
        try:
            os.execvp(executable, [executable, implementation] + sys.argv[1:])
        except OSError:
            pass

    sys.stderr.write(
        "ERROR: expand_dft_jobs.py requires Python 3. "
        "Set PYTHON3=/path/to/python3 or run python3 expand_dft_jobs.py.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
