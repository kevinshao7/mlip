from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution,
)
from tqdm import tqdm

import numpy as np
import os
import time

from mace.calculators import MACECalculator


def set_water_density(atoms, density_g_cm3):
    total_mass_amu = atoms.get_masses().sum()
    total_mass_g = total_mass_amu * 1.66053906660e-24

    volume_cm3 = total_mass_g / density_g_cm3
    volume_A3 = volume_cm3 * 1e24 #volume in cubic anstroms

    L = volume_A3 ** (1 / 3)

    atoms.set_cell([L, L, L], scale_atoms=True)
    atoms.set_pbc([True, True, True])
    atoms.wrap()

    return L
def simpleMD(init_conf, temp, calc, fname, s, T): 
    #s is save interval, T is total frames
    init_conf.calc = calc

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = VelocityVerlet( #velocityverlet is NVE integration
        init_conf,
        timestep=0.1 * units.fs,
    )

    if os.path.exists(fname):
        os.remove(fname)

    # arrays for plotting
    times = []
    temperatures = []
    pressures = []
    energies = []

    pbar = tqdm(total=T, desc=f"NVE MD initialized at {temp} K")

    def write_frame():
        atoms = dyn.atoms
        atoms.write(fname, append=True)

        t_fs = dyn.get_time() / units.fs

        E = atoms.get_potential_energy() / len(atoms)
        Tnow = atoms.get_temperature()

        try:
            stress = atoms.get_stress()
            Pnow = -np.mean(stress[:3]) / units.GPa
        except Exception:
            Pnow = np.nan

        density = (
            atoms.get_masses().sum()
            * 1.66053906660e-24
            / (atoms.get_volume() * 1e-24)
        )

        times.append(t_fs)
        temperatures.append(Tnow)
        pressures.append(Pnow)
        energies.append(E)

        pbar.update(s)
        pbar.set_postfix({
            "T(K)": f"{Tnow:.0f}",
            "P(GPa)": f"{Pnow:.2f}",
            "rho": f"{density:.3f}",
            "E(eV/a)": f"{E:.4f}",
        })

    dyn.attach(write_frame, interval=s)

    t0 = time.time()
    dyn.run(T)
    t1 = time.time()

    pbar.close()

    data = np.column_stack([
        times,
        temperatures,
        pressures,
        energies,
    ])

    npyname = fname.replace(".xyz", "_thermo.npy")
    txtname = fname.replace(".xyz", "_thermo.txt")

    np.save(npyname, data)
    np.savetxt(
        txtname,
        data,
        header="time_fs temperature_K pressure_GPa energy_eV_per_atom",
    )

    print(f"MD finished in {(t1 - t0) / 60:.2f} minutes")
    print(f"Trajectory written to {fname}")
    print(f"Thermo data written to {npyname}")
    print(f"Text data written to {txtname}")


init_conf = read(
    "ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz",
    index=0,
).copy()

# Choose density corresponding to your guessed 10 GPa water state
# You may need to scan this. Start e.g. 1.5–2.0 g/cm^3 for compressed water.

densitygcm3 = 1.5
L = set_water_density(init_conf, density_g_cm3=densitygcm3)
print(f"Set cubic box length to {L:.3f} Å")
print(f"Volume = {init_conf.get_volume():.3f} Å^3")

init_conf.info["charge"] = 0
init_conf.info["spin"] = 1
init_conf.info["external_field"] = [0.0, 0.0, 0.0]

mace_calc = MACECalculator(
    model_paths=["checkpoints/polar_ft_1m_run-123.model"],
    device="cuda",
    default_dtype="float32",
)

simpleMD(
    init_conf,
    temp=1500,
    calc=mace_calc,
    fname=f"mace_1500K_density_{densitygcm3}.xyz",
    s=10,
    T=5000,
)
