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
from mdinterface import SimCell
from mdinterface.database import Water
from mdinterface.core.specie import Specie


from mace.calculators import mace_polar


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "MDresults")

#generate initial configuration
water = Water()
boxsize=20
simbox = SimCell(xysize=[20, 20])
amm = Specie("NH3", name="NH3")
densitygcm3 = 1.0 #gcm3
simbox.add_solvent([water,amm],ratio=[7,1], zdim=20, density=densitygcm3)
simbox.build(padding=0.5)

atoms = simbox.to_ase()    
init_conf =atoms

# Choose density corresponding to your guessed 10 GPa water state
# You may need to scan this. Start e.g. 1.5–2.0 g/cm^3 for compressed water.

init_conf.info["charge"] = 0
init_conf.info["spin"] = 1
init_conf.info["external_field"] = [0.0, 0.0, 0.0]



def simpleMD(init_conf, temp, calc, fname, s, T): 
    #s is save interval, T is total frames
    init_conf.calc = calc

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = VelocityVerlet( #velocityverlet is NVE integration
        init_conf,
        timestep=0.01 * units.fs,
    )

    output_dir = os.path.dirname(fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(fname):
        os.remove(fname)

    # arrays for plotting
    times = []
    temperatures = []
    # pressures = []
    energies = []
    kinetic_energies = []
    total_energies = []

    pbar = tqdm(total=T, desc=f"NVE MD initialized at {temp} K")

    def write_frame():
        atoms = dyn.atoms
        atoms.write(fname, append=True)

        t_fs = dyn.get_time() / units.fs

        E = atoms.get_potential_energy() / len(atoms)
        Ekin = atoms.get_kinetic_energy() / len(atoms)
        Etot = E + Ekin
        Tnow = atoms.get_temperature()

        # try:
        #     stress = atoms.get_stress()
        #     Pnow = -np.mean(stress[:3]) / units.GPa
        # except Exception:
        #     Pnow = np.nan

        density = (
            atoms.get_masses().sum()
            * 1.66053906660e-24
            / (atoms.get_volume() * 1e-24)
        )

        times.append(t_fs)
        temperatures.append(Tnow)
        # pressures.append(Pnow)
        energies.append(E)
        kinetic_energies.append(Ekin)
        total_energies.append(Etot)

        # pbar.update(s)
        pbar.set_postfix({
            "T(K)": f"{Tnow:.0f}",
            # "P(GPa)": f"{Pnow:.2f}",
            "rho": f"{density:.3f}",
            "E(eV/a)": f"{E:.4f}",
            "Etot(eV/a)": f"{Etot:.4f}",
        })
    def update_progress():
        pbar.update(1)

    dyn.attach(update_progress, interval=1)
    dyn.attach(write_frame, interval=s)

    t0 = time.time()
    dyn.run(T)
    t1 = time.time()

    pbar.close()

    data = np.column_stack([
        times,
        temperatures,
        # pressures,
        energies,
        kinetic_energies,
        total_energies,
    ])

    npyname = fname.replace(".xyz", "_thermo.npy")
    txtname = fname.replace(".xyz", "_thermo.txt")

    np.save(npyname, data)
    np.savetxt(
        txtname,
        data,
        header=(
            "time_fs temperature_K pressure_GPa "
            "energy_eV_per_atom kinetic_energy_eV_per_atom "
            "total_energy_eV_per_atom"
        ),
    )

    print(f"MD finished in {(t1 - t0) / 60:.2f} minutes")
    print(f"Trajectory written to {fname}")
    print(f"Thermo data written to {npyname}")
    print(f"Text data written to {txtname}")




mace_calc = mace_polar(
    model="polar-1-s",
    device="cpu",
    default_dtype="float32",  # faster for MD
)

simpleMD(
    init_conf,
    temp=150,
    calc=mace_calc,
    fname=os.path.join(MD_RESULTS_DIR, f"mace_1500K_density_{densitygcm3}.xyz"),
    s=200,
    T=1000,
)
