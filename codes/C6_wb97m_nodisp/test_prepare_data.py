"""Scientific regression checks using a preserved real ORCA output."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase.io import read, write

from prepare_data import ROOT, HARTREE, BOHR, convert, verify


class ConversionTests(unittest.TestCase):
    source = ROOT / "outputsfull/acquisition/outputs/cluster_010.out"

    def test_known_energy_force_and_roundtrip(self):
        atoms = convert(self.source)
        self.assertAlmostEqual(atoms.info["REF_energy"], (-438.585767442637 + 0.006295804) * HARTREE)
        self.assertAlmostEqual(atoms.arrays["REF_forces"][0, 0], -(-0.009577013 - 0.000092074) * HARTREE / BOHR)
        self.assertFalse(atoms.pbc.any())
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data.extxyz"
            write(target, atoms)
            verify(read(target))

    def rejected(self, old, new):
        text = self.source.read_text(encoding="utf-8")
        self.assertIn(old, text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.out"
            path.write_text(text.replace(old, new), encoding="utf-8")
            with self.assertRaises(ValueError):
                convert(path)

    def test_charge_mismatch(self):
        self.rejected("* xyz 0 1", "* xyz 1 1")

    def test_spin_mismatch(self):
        self.rejected("* xyz 0 1", "* xyz 0 3")

    def test_missing_dispersion_forces(self):
        self.rejected("DISPERSION GRADIENT", "REMOVED GRADIENT")

    def test_failed_job(self):
        self.rejected("ORCA TERMINATED NORMALLY", "ORCA FAILED")


if __name__ == "__main__":
    unittest.main()
