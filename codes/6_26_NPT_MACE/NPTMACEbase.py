import os

from ase import units
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution,
)
from tqdm import tqdm

import numpy as np
import time

import torch

from mdinterface import SimCell
from mdinterface.database import Water
from mdinterface.core.specie import Specie

from mace.calculators import mace_polar


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputsfull")
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "outputsfull", ".cache"))

#generate initial configuration

densitygcm3 = 0.2 #gcm3, always 0.2 and allow NPT to naturally bring up pressure
pressuregpa = 1.0 # GPa
targetmolecules = 100
moleculemass = 18 #grams per mol
tempramptime = 10*1000*units.fs
MDtimestep = 0.5*units.fs
totaltimesteps = 50000
saveinterval =5
T_initial=300 #always keep at room temperature
T_final = 2500  # Heating target for downstream runs; not used during pressure equilibration.

if saveinterval <= 0:
    raise ValueError(f"saveinterval must be positive, got {saveinterval}")

RUN_SEED = int.from_bytes(os.urandom(8), "little") % 1_000_000_000
np.random.seed(RUN_SEED)
print(f"RUN_SEED {RUN_SEED}")
print(f"saveinterval_steps {saveinterval}")

NA = 6.022e23
boxsize=(((targetmolecules*moleculemass/NA)/densitygcm3)**(1/3))*1e8 #boxsize in angstroms
water = Water()
simbox = SimCell(xysize=[boxsize, boxsize])
amm = Specie("NH3", name="NH3")

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


def simpleMD(init_conf, temp, pressure_gpa, calc, fname, checkpoint_fname, s, T, T_thermo=100):
    # s is save interval, T is total NPT integration steps
    init_conf.calc = calc

    MaxwellBoltzmannDistribution(init_conf, temperature_K=T_initial)
    Stationary(init_conf)
    ZeroRotation(init_conf)
    # ----------------------------
    # 1. Brief NVT thermalization
    # ----------------------------
    thermo = Langevin(
        init_conf,
        timestep=MDtimestep,
        temperature_K=T_initial, #Start near room temperature
        friction=0.01 / units.fs,   # damping time ~100 fs
    )

    print(f"Initial NVT Langevin thermalization for {T_thermo} steps at {T_initial} K...")
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

    output_dir = os.path.dirname(fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(fname):
        os.remove(fname)

    # arrays for plotting; pressure equilibration is kept at fixed temperature
    times = []
    temperatures = []
    pressures = []
    energies = []
    kinetic_energies = []
    total_energies = []
    target_temperatures = []

    pbar = tqdm(
        total=T,
        desc=f"NPT pressure equilibration at {temp} K and {pressure_gpa} GPa",
    )



    def make_npt_dynamics(temperature_K):
        return IsotropicMTKNPT(
            init_conf,
            timestep=MDtimestep,
            temperature_K=temperature_K,
            pressure_au=pressure_au,
            tdamp=100 * units.fs,
            pdamp=1000 * units.fs
        )

    def write_frame(dyn, t_fs, target_temperature):
        atoms = dyn.atoms
        atoms.write(fname, append=True)

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
        target_temperatures.append(target_temperature)

        # pbar.update(s)
        pbar.set_postfix({
            "T(K)": f"{Tnow:.0f}",
            "Ttarget(K)": f"{target_temperature:.0f}",
            "P(GPa)": f"{Pnow:.2f}",
            "rho": f"{density:.3f}",
            "E(eV/a)": f"{E:.4f}",
            "Etot(eV/a)": f"{Etot:.4f}",
        })

    t0 = time.time()
    dyn = make_npt_dynamics(temp)

    def update_progress():
        pbar.update(1)

    def write_production_frame():
        if dyn.get_time() == 0:
            return
        t_fs = dyn.get_time() / units.fs
        write_frame(dyn, t_fs=t_fs, target_temperature=temp)

    dyn.attach(update_progress, interval=1)
    dyn.attach(write_production_frame, interval=s)

    dyn.run(T)
    t1 = time.time()

    pbar.close()
    dyn.atoms.write(checkpoint_fname)

    data = np.column_stack([
        times,
        temperatures,
        pressures,
        energies,
        kinetic_energies,
        total_energies,
        target_temperatures,
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
            "total_energy_eV_per_atom target_temperature_K"
        ),
    )

    print(f"MD finished in {(t1 - t0) / 60:.2f} minutes")
    print(f"Trajectory written to {fname}")
    print(f"Final checkpoint written to {checkpoint_fname}")
    print(f"Thermo data written to {npyname}")
    print(f"Text data written to {txtname}")




MACE_DEVICE = os.environ.get("MLIP_MACE_DEVICE", "cuda")
if MACE_DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        "MLIP_MACE_DEVICE is set to CUDA, but PyTorch cannot see an NVIDIA CUDA GPU. "
        "Run on a GPU node or set MLIP_MACE_DEVICE=cpu explicitly for a CPU test."
    )

mace_calc = mace_polar(
    model="polar-1-s",
    device=MACE_DEVICE,
    default_dtype="float32",  # faster for MD
)

simpleMD(
    init_conf,
    temp=T_initial,
    pressure_gpa=pressuregpa,
    calc=mace_calc,
    fname=os.path.join(
        MD_RESULTS_DIR,
        f"pressure_equil_seed_{RUN_SEED}_P_{pressuregpa:g}GPa_T_{T_initial:g}K_density_{densitygcm3}.xyz",
    ),
    checkpoint_fname=os.path.join(
        MD_RESULTS_DIR,
        f"checkpoint_seed_{RUN_SEED}_P_{pressuregpa:g}GPa_T_{T_initial:g}K_density_{densitygcm3}.xyz",
    ),
    s=saveinterval,
    T=totaltimesteps
)
