import os
# -------------------------
# CPU parallelism settings
# -------------------------
N_THREADS = "20"

os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["TORCH_NUM_THREADS"] = N_THREADS

# Optional: avoid oversubscription from nested OpenMP regions
os.environ["OMP_PROC_BIND"] = "spread"
os.environ["OMP_PLACES"] = "threads"

from ase.io import read
from ase import units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution,
)
from tqdm import tqdm

import numpy as np
import time

from mdinterface import SimCell
from mdinterface.database import Water
from mdinterface.core.specie import Specie

import torch
torch.set_num_threads(int(N_THREADS))
torch.set_num_interop_threads(1)

from mace.calculators import mace_polar


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "MDresults")

#generate initial configuration
water = Water()
simbox = SimCell(xysize=[boxsize, boxsize])
amm = Specie("NH3", name="NH3")
densitygcm3 = 1.0 #gcm3
pressuregpa = 1.0 # GPa
targetmolecules = 100
moleculemass = 18 #grams per mol
NA = 6.022e23
boxsize=(((targetmolecules*moleculemass/NA)/densitygcm3)**(1/3))*1e8 #boxsize in angstroms


simbox.add_solvent([water,amm],ratio=[7,1], zdim=boxsize, density=densitygcm3)
simbox.build(padding=0.5)

atoms = simbox.to_ase()    
init_conf =atoms
print("Number of atoms:", len(init_conf))
print("Chemical formula:", init_conf.get_chemical_formula())
# Choose density corresponding to your guessed 10 GPa water state
# You may need to scan this. Start e.g. 1.5–2.0 g/cm^3 for compressed water.

init_conf.info["charge"] = 0
init_conf.info["spin"] = 1
init_conf.info["external_field"] = [0.0, 0.0, 0.0]

# Berendsen NPT needs a compressibility.  This ambient-water value is only a
# numerical barostat parameter here, not a claim about Uranus-interior water.
WATER_COMPRESSIBILITY_AU = 4.57e-5 / units.bar


def simpleMD(init_conf, temp, pressure_gpa, calc, fname, s, T, T_thermo=100):
    # s is save interval, T is total NPT integration steps
    init_conf.calc = calc

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp)
    Stationary(init_conf)
    ZeroRotation(init_conf)
    # ----------------------------
    # 1. Brief NVT thermalization
    # ----------------------------
    thermo = Langevin(
        init_conf,
        timestep=0.1 * units.fs,
        temperature_K=temp,
        friction=0.01 / units.fs,   # damping time ~100 fs
    )

    print(f"Initial NVT Langevin thermalization for {T_thermo} steps at {temp} K...")
    starttime = time.time()
    thermo.run(T_thermo)
    endtime=time.time()
    print("steptime "+str(endtime-starttime))
    # Remove any drift after      thermostatting
    Stationary(init_conf)
    ZeroRotation(init_conf)
    # ----------------------
    # 2. NPT production run
    # ----------------------
    pressure_au = pressure_gpa * units.GPa
    dyn = NPTBerendsen(
        init_conf,
        timestep=0.1 * units.fs,
        temperature_K=temp,
        pressure_au=pressure_au,
        taut=100 * units.fs,
        taup=1000 * units.fs,
        compressibility_au=WATER_COMPRESSIBILITY_AU,
    )

    output_dir = os.path.dirname(fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(fname):
        os.remove(fname)

    # arrays for plotting
    times = []
    temperatures = []
    pressures = []
    energies = []
    kinetic_energies = []
    total_energies = []

    pbar = tqdm(total=T, desc=f"NPT MD at {temp} K and {pressure_gpa} GPa")

    def write_frame():
        atoms = dyn.atoms
        atoms.write(fname, append=True)

        t_fs = dyn.get_time() / units.fs

        E = atoms.get_potential_energy() / len(atoms)
        Ekin = atoms.get_kinetic_energy() / len(atoms)
        Etot = E + Ekin
        Tnow = atoms.get_temperature()

        try:
            stress = atoms.get_stress(include_ideal_gas=True)
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
        kinetic_energies.append(Ekin)
        total_energies.append(Etot)

        # pbar.update(s)
        pbar.set_postfix({
            "T(K)": f"{Tnow:.0f}",
            "P(GPa)": f"{Pnow:.2f}",
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
        pressures,
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
    temp=temp,
    pressure_gpa=pressuregpa,
    calc=mace_calc,
    fname=os.path.join(MD_RESULTS_DIR, f"mace_1500K_density_{densitygcm3}.xyz"),
    s=10,
    T=1000,
)
