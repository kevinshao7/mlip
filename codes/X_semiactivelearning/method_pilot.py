"""Verify direct dispersion-free ORCA against an audited C6 geometry/label."""
from pathlib import Path


def prepare(args):
    import campaign
    from ase.io import read, write
    from site_dft import prepare as prepare_sites
    source = campaign.MLIP / 'outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/wb97m.extxyz'
    atoms = read(source, index=0)
    if atoms.info.get('dft_functional_basis') != 'wB97M/def2-TZVPD':
        raise ValueError('Pilot reference is not wB97M-only')
    charge = campaign.formal_charge(atoms.get_chemical_symbols())
    if atoms.info.get('charge') != charge or atoms.info.get('spin') != 1:
        raise ValueError('Pilot reference charge/spin mismatch')
    ddir = campaign.OUT / 'method_pilot'
    if ddir.exists():
        raise FileExistsError(f'Preserve existing pilot: {ddir}')
    inputs = ddir / 'inputs'; inputs.mkdir(parents=True)
    route = campaign.config()['dft']['functional_basis']
    (inputs / 'pilot.inp').write_text(campaign.dft_input(atoms, charge, 1, campaign.config()['dft']['orca_nprocs'], route))
    write(ddir / 'reference.extxyz', atoms)
    launch = campaign.ROOT / 'dft_launches' / campaign.RUN_ID / 'method_pilot'
    (launch / 'inputs').mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copy2(inputs / 'pilot.inp', launch / 'inputs/pilot.inp')
    prepare_sites(ddir, launch, [{'stem':'pilot'}], campaign.config()['dft'], campaign)
    campaign.write_json(ddir / 'reference.json', {'source': str(source.relative_to(campaign.MLIP)), 'source_sha256': campaign.sha(source), 'frame':0, 'reference_sha256':campaign.sha(ddir / 'reference.extxyz'), 'charge':charge, 'spin':1})
    campaign.event('method_pilot_prepared', None, [source], [ddir / 'reference.json', inputs / 'pilot.inp'])
    print(f'Submit the single-geometry pilot on Viper or BlueHive: {launch}')


def verify(args):
    import campaign
    import json
    import numpy as np
    from ase.io import read
    from orca_output_validation import Job, process
    ddir = campaign.OUT / 'method_pilot'
    reference = read(ddir / 'reference.extxyz')
    manifest = json.loads((ddir / 'reference.json').read_text())
    if campaign.sha(ddir / 'reference.extxyz') != manifest['reference_sha256']:
        raise ValueError('Pilot reference changed')
    charge = campaign.formal_charge(reference.get_chemical_symbols())
    if reference.info.get('charge') != charge:
        raise ValueError('Pilot reference charge mismatch')
    result, atoms = process(Job(0, 'pilot', ddir / 'inputs/pilot.inp', ddir / 'outputs/pilot.out', charge, 1, len(reference), campaign.config()['dft']['functional_basis']))
    if atoms is None:
        raise ValueError(f'Pilot failed strict validation: {result}')
    np.testing.assert_allclose(atoms.positions, reference.positions, rtol=0, atol=1e-8)
    if list(atoms.symbols) != list(reference.symbols):
        raise ValueError('Pilot atom order changed')
    energy_error = abs(atoms.info['REF_energy']-reference.info['REF_energy'])
    force_error = float(np.max(np.abs(atoms.arrays['REF_forces']-reference.arrays['REF_forces'])))
    # Numerical integration/printed gradients can differ across ORCA builds.
    # These are explicit pilot equivalence tolerances, not the advancement gate.
    if energy_error > 1e-3 or force_error > 1e-4:
        raise ValueError(f'Direct/C6 disagreement: {energy_error} eV, {force_error} eV/Angstrom')
    sources = [Path(__file__), campaign.ROOT / 'orca_output_validation.py', campaign.ROOT / 'workflow_policy.py', campaign.ROOT / 'campaign.py']
    record = {'method':campaign.config()['dft']['functional_basis'], 'energy_error_ev':energy_error, 'max_force_component_error_ev_per_angstrom':force_error, 'output_sha256':campaign.sha(ddir / 'outputs/pilot.out'), 'input_sha256':campaign.sha(ddir / 'inputs/pilot.inp'), 'validator_sha256':{p.name:campaign.sha(p) for p in sources}}
    campaign.write_json(campaign.OUT / 'method_verified.json', record)
    campaign.event('method_pilot_verified', None, [ddir / 'outputs/pilot.out'], [campaign.OUT / 'method_verified.json'], **record)
    print(record)
