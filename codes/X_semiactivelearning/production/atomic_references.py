"""Audit neutral atomic E0 evidence before campaign training."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARTREE_TO_EV = 27.211386245988


def prepare():
    # Neutral atoms are reference states; ground-state multiplicities are
    # independent of the formal-charge convention for molecular training data.
    evidence = ROOT.parent / 'C5_atomizationD4rev/runs'
    references = {}; sources = {}
    import campaign
    for symbol, number, spin in [('H', 1, 2), ('N', 7, 4), ('O', 8, 3), ('S', 16, 3)]:
        source = evidence / symbol / f'orcaatomizationD4rev{symbol}.out'
        text = source.read_text()
        if 'ORCA TERMINATED NORMALLY' not in text:
            raise ValueError(f'Incomplete atomic reference: {source}')
        if not re.search(rf'\*\s*xyz\s+0\s+{spin}\s*\n\s*\|\s*\d+>\s*{symbol}\s', text, re.I):
            raise ValueError(f'Atomic input identity/charge/spin not verified: {source}')
        if not re.search(r'Total Charge\s+Charge\s+\.+\s+0\b', text):
            raise ValueError(f'Atomic charge mismatch: {source}')
        if not re.search(rf'Multiplicity\s+Mult\s+\.+\s+{spin}\b', text):
            raise ValueError(f'Atomic spin mismatch: {source}')
        if '! wB97M-D4rev def2-TZVPD EnGrad RIJCOSX' not in text:
            raise ValueError(f'Atomic method mismatch: {source}')
        dispersion = re.findall(r'Dispersion correction\s+([+-]?[\d.]+)\s*$', text, re.M)
        energies = re.findall(r'FINAL SINGLE POINT ENERGY\s+([+-]?[\d.]+)', text)
        if len(dispersion) != 1 or float(dispersion[0]) != 0 or len(energies) != 1:
            raise ValueError(f'Atomic dispersion does not vanish: {source}')
        norms = re.findall(r'Norm of the Dispersion gradient\s+\.\.\.\s+([+-]?[\d.]+)', text)
        if len(norms) != 1 or float(norms[0]) != 0:
            raise ValueError(f'Nonzero or missing atomic dispersion-gradient norm: {source}')
        references[str(number)] = float(energies[0]) * HARTREE_TO_EV
        sources[symbol] = {'path': str(source.relative_to(campaign.MLIP)), 'sha256': campaign.sha(source), 'charge': 0, 'multiplicity': spin, 'D4_energy_hartree': 0.0}
    destination = ROOT / 'atomic_references'
    destination.mkdir(exist_ok=True)
    content = json.dumps(references, indent=2) + '\n'
    path = destination / 'e0s_wb97m.json'
    if path.exists() and path.read_text() != content:
        raise ValueError('Existing atomic reference differs; preserve and investigate')
    path.write_text(content)
    campaign.write_json(destination / 'provenance.json', {'method': 'wB97M/def2-TZVPD', 'energy_units': 'eV', 'evidence': sources, 'e0s_sha256': campaign.sha(path), 'convention': 'neutral isolated atoms; printed D4 energy exactly zero'})
    return path


if __name__ == '__main__':
    print(prepare())
