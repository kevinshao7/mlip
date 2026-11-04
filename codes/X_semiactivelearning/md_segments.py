"""Restartable production within the DAIS 24 h allocation limit.

Only the prepared P10 batch uses this runner. Production uses one persistent
MTK integrator; checkpoints retain its extended thermostat/barostat state.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ase
from ase import units
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
import numpy as np

PAUSED_EXIT_CODE = 75
DYNAMICS_FIELDS = ('_q','_p','_eps','_p_eps','_cell0','_volume0')
THERMOSTAT_FIELDS = ('_eta','_p_eta')
BAROSTAT_FIELDS = ('_xi','_p_xi')


def save_restart(path, dyn, metadata):
    """Atomically replace a versioned-schema, pickle-free restart file."""
    metadata={**metadata,'ase_version':ase.__version__, 'schema':1,
              'atom_array_names':list(dyn.atoms.arrays), 'nsteps':dyn.nsteps}
    payload={'metadata':np.asarray(json.dumps(metadata)), 'cell':np.asarray(dyn.atoms.cell)}
    for name,values in dyn.atoms.arrays.items(): payload['atom_'+name]=values
    for name in DYNAMICS_FIELDS: payload['dyn_'+name]=np.asarray(getattr(dyn,name))
    for name in THERMOSTAT_FIELDS: payload['thermo_'+name]=np.asarray(getattr(dyn._thermostat,name))
    for name in BAROSTAT_FIELDS: payload['baro_'+name]=np.asarray(getattr(dyn._barostat,name))
    temporary=Path(str(path)+'.partial')
    with temporary.open('wb') as handle:
        np.savez(handle,**payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)


def load_restart(path, atoms, expected):
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata']))
        if meta['schema']!=1 or meta['ase_version']!=ase.__version__:
            raise ValueError('Restart schema/ASE version changed')
        for key,value in expected.items():
            if meta.get(key)!=value: raise ValueError(f'Restart setting changed: {key}')
        if not np.array_equal(saved['atom_numbers'],atoms.numbers):
            raise ValueError('Restart atom identity/order mismatch')
        for name in meta['atom_array_names']:
            atoms.set_array(name,saved['atom_'+name].copy())
        atoms.set_cell(saved['cell'],scale_atoms=False)
        state={name:saved[name].copy() for name in saved.files if name.startswith(('dyn_','thermo_','baro_'))}
    return meta,state


def restore_integrator(dyn, state, meta):
    for prefix,obj,fields in [('dyn_',dyn,DYNAMICS_FIELDS),('thermo_',dyn._thermostat,THERMOSTAT_FIELDS),('baro_',dyn._barostat,BAROSTAT_FIELDS)]:
        for name in fields:
            value=state[prefix+name]
            setattr(obj,name,value.item() if value.ndim==0 else value.copy())
    dyn.nsteps=meta['nsteps']; dyn._update_atoms()


def recover_output(path, committed_bytes):
    """Archive output beyond the last atomic checkpoint before replaying it."""
    if not path.exists():
        if committed_bytes: raise ValueError(f'Missing checkpointed output: {path}')
        return
    if path.stat().st_size<committed_bytes:
        raise ValueError(f'Checkpointed output was truncated externally: {path}')
    if path.stat().st_size>committed_bytes:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        with path.open('r+b') as handle:
            handle.seek(committed_bytes)
            tail=path.with_name(path.name+f'.uncommitted_{stamp}')
            with tail.open('xb') as archive: archive.write(handle.read())
            handle.truncate(committed_bytes)


def staged_md(init_conf,temp_final,pressure_gpa,calc,fname,checkpoint_fname,s,T,T_thermo=100,
              timestep_fs=.5,pressure_ramp_ps=2.,temperature_ramp_ps=2.):
    root=Path(fname).parent; root.mkdir(parents=True,exist_ok=True)
    trajectory=Path(fname); thermo=trajectory.with_name(trajectory.stem+'_thermo.txt')
    restart=root/'restart.npz'
    ramp_p=round(pressure_ramp_ps*1000/timestep_fs)
    ramp_t=round(temperature_ramp_ps*1000/timestep_fs)
    ramps=ramp_p+ramp_t; production=T-ramps
    if production<1 or production%s: raise ValueError('Production length must be a positive multiple of the save interval')
    expected={'temperature_K':temp_final,'pressure_gpa':pressure_gpa,'timestep_fs':timestep_fs,
              'pressure_ramp_steps':ramp_p,'temperature_ramp_steps':ramp_t,
              'total_steps':T,'save_interval_steps':s,
              'model_sha256':os.environ.get('MLIP_MODEL_SHA256'),
              'batch_config_sha256':os.environ.get('MLIP_BATCH_CONFIG_SHA256'),
              'torch_version':os.environ.get('MLIP_TORCH_VERSION'), 'pbc':init_conf.pbc.tolist()}
    init_conf.calc=calc
    step=0; resumed=None; state=None; last_stage='pressure_ramp'
    if restart.exists():
        resumed,state=load_restart(restart,init_conf,expected)
        step=resumed['step']; last_stage=resumed['stage']
        recover_output(trajectory,resumed['trajectory_bytes'])
        recover_output(thermo,resumed['thermo_bytes'])
    else:
        if trajectory.exists() or thermo.exists(): raise FileExistsError('Uncheckpointed output exists; inspect before starting')
        MaxwellBoltzmannDistribution(init_conf,temperature_K=300)
        Stationary(init_conf); ZeroRotation(init_conf)
        thermal=Langevin(init_conf,timestep=timestep_fs*units.fs,temperature_K=300,friction=.01/units.fs)
        thermal.run(T_thermo); Stationary(init_conf); ZeroRotation(init_conf)
    def make(temp,pressure):
        return IsotropicMTKNPT(init_conf,timestep=timestep_fs*units.fs,temperature_K=temp,
                               pressure_au=pressure*units.GPa,tdamp=100*units.fs,pdamp=1000*units.fs)
    dyn=None
    if step>=ramps:
        dyn=make(temp_final,pressure_gpa)
        if resumed and last_stage=='production_hold': restore_integrator(dyn,state,resumed)
    elif resumed:
        # Ramp chunks retain the existing changing-target schedule and begin
        # from the checkpointed positions/momenta, without rethermalization.
        pass
    def checkpoint():
        meta={**expected,'step':step,'stage':last_stage,
              'trajectory_bytes':trajectory.stat().st_size if trajectory.exists() else 0,
              'thermo_bytes':thermo.stat().st_size if thermo.exists() else 0}
        save_restart(restart,dyn,meta)
        status={**meta,'production_steps':max(0,step-ramps),
                'production_time_ps':max(0,step-ramps)*timestep_fs/1000,'complete':step==T}
        (root/'production_status.json').write_text(json.dumps(status,indent=2)+'\n')
    started=time.monotonic(); budget=float(os.environ.get('MLIP_SEGMENT_SECONDS',22*3600))
    checkpoint_interval=1000
    while step<T:
        if step<ramp_p:
            last_stage='pressure_ramp'; fraction=step/max(ramp_p-1,1)
            dyn=make(300,.0001+fraction*(pressure_gpa-.0001))
            count=min(20,ramp_p-step)
        elif step<ramps:
            last_stage='temperature_ramp'; fraction=(step-ramp_p)/max(ramp_t-1,1)
            dyn=make(300+fraction*(temp_final-300),pressure_gpa)
            count=min(20,ramps-step)
        else:
            if dyn is None or last_stage!='production_hold': dyn=make(temp_final,pressure_gpa)
            last_stage='production_hold'
            count=min(20,T-step,s-((step-ramps)%s))
        dyn.run(count); step+=count
        if step>ramps and (step-ramps)%s==0:
            init_conf.write(trajectory,append=True)
            pressure=-float(np.mean(init_conf.get_stress(include_ideal_gas=True)[:3]))/units.GPa
            empty=not thermo.exists() or thermo.stat().st_size==0
            with thermo.open('a') as handle:
                if empty: handle.write('# time_fs temperature_K pressure_GPa volume_A3\n')
                handle.write(f'{step*timestep_fs:.12g} {init_conf.get_temperature():.12g} {pressure:.12g} {init_conf.get_volume():.12g}\n')
        if step%checkpoint_interval==0 or step==T or time.monotonic()-started>=budget:
            checkpoint()
        if step<T and time.monotonic()-started>=budget:
            print(f'Segment paused at production time {max(0,step-ramps)*timestep_fs/1000:g} ps; exact restart saved',flush=True)
            raise SystemExit(PAUSED_EXIT_CODE)
    checkpoint()
    init_conf.write(checkpoint_fname)
    print(f'Production complete: {production*timestep_fs/1000:g} ps, {production//s} saved frames',flush=True)
