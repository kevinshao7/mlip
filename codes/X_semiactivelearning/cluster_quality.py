"""Whole-molecule PBC placement and auditable isolated-cluster checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic


def molecule_images(frame, components, centre, allowed):
    """Unwrap each molecule internally, then translate its complete image.

    An atomwise MIC relative to a distant centre can split a molecule at the
    half-cell surface. A common translation for its internally unwrapped atoms
    preserves all molecular pair distances.
    """
    anchor = frame.positions[components[centre][0]]
    central_local, _ = find_mic(frame.positions[components[centre]]-anchor, frame.cell, pbc=True)
    central_centroid = central_local.mean(axis=0)
    positions = {}; centroids = {}; max_error = 0.
    for ci in allowed:
        indices = components[ci]
        first = frame.positions[indices[0]]
        local, _ = find_mic(frame.positions[indices]-first, frame.cell, pbc=True)
        offset = local.mean(axis=0)
        delta, _ = find_mic(first-anchor+offset-central_centroid, frame.cell, pbc=True)
        centroids[ci] = central_centroid+delta
        positions[ci] = local-offset+centroids[ci]
        source_delta = frame.positions[indices][:,None,:]-frame.positions[indices][None,:,:]
        _, source_distances = find_mic(source_delta.reshape(-1,3), frame.cell, pbc=True)
        isolated = np.linalg.norm(positions[ci][:,None,:]-positions[ci][None,:,:],axis=-1).ravel()
        error = float(np.max(np.abs(isolated-source_distances)))
        max_error = max(max_error,error)
        if error>1e-8:
            raise ValueError(f'Molecule {ci} cannot be unwrapped without changing internal distances: {error} Angstrom')
    return positions, centroids, max_error


def audit_cluster(atoms):
    from campaign import formal_charge
    allowed = {'H2O','HO','H3O','H3N','H2N','H4N'}
    if atoms.pbc.any() or np.any(atoms.cell):
        raise ValueError('DFT clusters must be isolated: PBC=False, no cell')
    if not np.isfinite(atoms.positions).all():
        raise ValueError('Nonfinite cluster geometry')
    sizes = list(atoms.info.get('molecule_atom_counts', []))
    formulas = str(atoms.info.get('selected_formulas','')).split(';')
    if len(sizes)!=6 or len(formulas)!=6 or sum(sizes)!=len(atoms):
        raise ValueError('Expected six complete molecules with preserved membership')
    if not atoms.info.get('whole_molecules_preserved'):
        raise ValueError('Missing whole-molecule PBC audit')
    if atoms.info['max_internal_distance_error_angstrom']>1e-8:
        raise ValueError('Intramolecular geometry changed during extraction')
    offset = 0
    from collections import Counter
    for size, expected in zip(sizes, formulas):
        counts=Counter(atoms.symbols[offset:offset+size]); offset+=size
        formula=''.join(symbol+(str(counts[symbol]) if counts[symbol]>1 else '') for symbol in sorted(counts))
        if formula!=expected or formula not in allowed:
            raise ValueError('Molecule membership/formula mismatch')
    charge = formal_charge(atoms.get_chemical_symbols())
    if atoms.info.get('charge')!=charge or atoms.info.get('spin')!=1 or atoms.info.get('multiplicity')!=1:
        raise ValueError('Formal charge or multiplicity mismatch')
    distances=atoms.get_all_distances(mic=False)
    distances=distances[np.triu_indices(len(atoms),k=1)]
    minimum=float(distances.min())
    if minimum<=1e-8:
        raise ValueError('Coincident atoms in DFT cluster')
    # Report the observed minimum; do not silently impose a new physical cutoff.
    return {'minimum_pair_distance_angstrom':minimum,
            'max_internal_distance_error_angstrom':float(atoms.info['max_internal_distance_error_angstrom']),
            'charge':charge,'molecules':6,'atoms':len(atoms)}


def audit_sets(paths, summary=None):
    from ase.io import iread
    from campaign import sha
    keys=set(); rows=[]
    for role,path in paths.items():
        for index,atoms in enumerate(iread(path)):
            key=(atoms.info['source_trajectory_id'],int(atoms.info['source_frame']),int(atoms.info['central_allowed_component']))
            from workflow_policy import partition
            if (partition(key[1])=='evaluation') != (role=='evaluation'):
                raise ValueError('Acquisition/evaluation frame partition mismatch')
            if summary is not None:
                start=int(summary.get('production_start_frame_0based',0))
                stride=int(summary.get('production_frame_stride',1))
                samples=summary['replica_summaries'][key[0]]['frames']
                if key[1]<start or (key[1]-start)%stride or (key[1]-start)//stride>=samples:
                    raise ValueError('DFT cluster lies outside the reviewed analysis window')
            if key in keys:
                raise ValueError(f'Duplicate frame/centre source across DFT datasets: {key}')
            keys.add(key)
            rows.append({'role':role,'index':index,**audit_cluster(atoms)})
    return {'status':'passed','total':len(rows),'input_sha256':{role:sha(path) for role,path in paths.items()},
            'minimum_pair_distance_angstrom':min(row['minimum_pair_distance_angstrom'] for row in rows),
            'maximum_internal_distance_error_angstrom':max(row['max_internal_distance_error_angstrom'] for row in rows),
            'clusters':rows,'short_contact_policy':'report distances; no new physical cutoff imposed'}
