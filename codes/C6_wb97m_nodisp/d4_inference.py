"""Explicit D4rev ASE calculator and validation against extracted ORCA labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import iread

# ORCA 6.1 manual, Table 3.15: wB97M-D4rev (not the older wB97M-D4).
D4REV = dict(s6=1.0, s8=0.842, a1=0.359, a2=4.668, s9=1.0, alp=16.0)


def d4_calculator(charge):
    """Charge is the explicitly supplied total system charge, not partial charges.

    ASE supplies the current cell/PBC to DFTD4; periodic deployment must use
    its actual cell/PBC, whereas the extracted training clusters are isolated.
    """
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
    from dftd4.interface import DampingParam, DispersionModel

    class ExplicitChargeD4(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            system = self.atoms
            # Pass total charge directly; do not depend on ASE initial partial charges.
            model = DispersionModel(system.numbers, system.positions / Bohr,
                                    charge=float(charge), lattice=system.cell.array / Bohr,
                                    periodic=system.pbc, model="d4")
            result = model.get_dispersion(DampingParam(**D4REV), grad=True)
            self.results.update(energy=float(result["energy"]) * Hartree,
                                free_energy=float(result["energy"]) * Hartree,
                                forces=-result["gradient"] * Hartree / Bohr)
            if system.pbc.any():
                stress = result["virial"] * Hartree / system.get_volume()
                self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

    return ExplicitChargeD4()


def with_d4(ml_calculator, *, charge):
    """Use only with an ML model trained on the dispersion-free labels."""
    from ase.calculators.mixing import SumCalculator
    return SumCalculator([ml_calculator, d4_calculator(charge)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--energy-atol", type=float, default=1e-6, help="eV per cluster")
    parser.add_argument("--force-atol", type=float, default=1e-6, help="eV/Angstrom per component")
    args = parser.parse_args()
    energy_errors, force_errors = [], []
    for atoms in iread(args.dataset):
        reference_energy = atoms.info["D4_energy"]
        reference_forces = atoms.arrays["D4_forces"].copy()
        atoms.calc = d4_calculator(atoms.info["charge"])
        energy_errors.append(abs(atoms.get_potential_energy() - reference_energy))
        force_errors.append(float(np.max(abs(atoms.get_forces() - reference_forces))))
    if not energy_errors:
        raise SystemExit("Empty dataset")
    result = dict(frames=len(energy_errors), max_energy_error_eV=max(energy_errors),
                  max_force_error_eV_per_A=max(force_errors), parameters=D4REV)
    print(json.dumps(result, indent=2))
    if max(energy_errors) > args.energy_atol or max(force_errors) > args.force_atol:
        raise SystemExit("D4 backend differs from ORCA: resolve version/model/cutoff differences before inference")


if __name__ == "__main__":
    main()
