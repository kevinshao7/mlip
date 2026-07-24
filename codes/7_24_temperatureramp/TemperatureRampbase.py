import os

from ase import units
from ase.io import read
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

from mace.calculators import mace_polar


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputsfull", "temperature_ramp")
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "outputsfull", ".cache"))

# Default is a selected r09 hot pure-water pressure-equilibration trajectory.
# Override CHECKPOINT_XYZ for other pressure/composition cases.
DEFAULT_CHECKPOINT_XYZ = (
    "/home/kevinsh/mlip/outputsfull/r09_hot_w/"
    "pressure_equil_seed_353168294_P_15GPa_T_300K_density_0.2.xyz"
)
INPUT_CHECKPOINT_XYZ = os.environ.get("CHECKPOINT_XYZ", DEFAULT_CHECKPOINT_XYZ)
CHECKPOINT_FRAME = int(os.environ.get("CHECKPOINT_FRAME", "-1"))

densitygcm3 = 0.2 # metadata from initial pressure-equilibration setup, g/cm3
pressuregpa = 1.0 # GPa
tempramptime = 100*1000*units.fs
holdtime = 100*1000*units.fs
MDtimestep = 0.5*units.fs
saveinterval =100
T_initial=300 # pressure-equilibrated checkpoint temperature, K
T_final = 2500  # Placeholder; TemperatureRampexpand.py overwrites this per Uranus row.

if not INPUT_CHECKPOINT_XYZ:
    raise ValueError(
        "Set CHECKPOINT_XYZ to the pressure-equilibrated checkpoint .xyz file "
        "that should be heated."
    )
if not os.path.exists(INPUT_CHECKPOINT_XYZ):
    raise FileNotFoundError(f"CHECKPOINT_XYZ does not exist: {INPUT_CHECKPOINT_XYZ}")
if saveinterval <= 0:
    raise ValueError(f"saveinterval must be positive, got {saveinterval}")

RUN_SEED = int.from_bytes(os.urandom(8), "little") % 1_000_000_000
np.random.seed(RUN_SEED)
print(f"RUN_SEED {RUN_SEED}")
print(f"saveinterval_steps {saveinterval}")
print(f"input_checkpoint {INPUT_CHECKPOINT_XYZ}")
print(f"checkpoint_frame {CHECKPOINT_FRAME}")

init_conf = read(INPUT_CHECKPOINT_XYZ, index=CHECKPOINT_FRAME)
init_conf.set_pbc([True, True, True])
if init_conf.get_volume() <= 0:
    raise ValueError(
        "Input checkpoint has no valid periodic cell. Use an extended XYZ "
        "written by ASE from the pressure-equilibration workflow."
    )

init_conf.info["charge"] = 0
init_conf.info["spin"] = 1
init_conf.info["external_field"] = [0.0, 0.0, 0.0]

print("Number of atoms:", len(init_conf))
print("Chemical formula:", init_conf.get_chemical_formula())
print("Cell volume A^3:", init_conf.get_volume())

rampsteps = int(tempramptime/MDtimestep)
holdsteps = int(holdtime/MDtimestep)
updateinterval = 20
print("rampsteps")
print(rampsteps)
print("holdsteps")
print(holdsteps)


def run_temperature_ramp(init_conf, temp_final, pressure_gpa, calc, fname, checkpoint_fname, s):
    init_conf.calc = calc

    # XYZ checkpoints do not preserve velocities; start the heating leg from a
    # fresh Maxwell-Boltzmann draw at the equilibrated checkpoint temperature.
    MaxwellBoltzmannDistribution(init_conf, temperature_K=T_initial)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    pressure_au = pressure_gpa * units.GPa

    output_dir = os.path.dirname(fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(fname):
        os.remove(fname)

    times = []
    temperatures = []
    pressures = []
    densities = []
    energies = []
    kinetic_energies = []
    total_energies = []
    target_temperatures = []

    pbar = tqdm(
        total=rampsteps + holdsteps,
        desc=(
            f"NPT temperature ramp {T_initial} -> {temp_final} K, "
            f"then hold at {temp_final} K and {pressure_gpa} GPa"
        ),
    )

    def make_npt_dynamics(temperature_K):
        return IsotropicMTKNPT(
            init_conf,
            timestep=MDtimestep,
            temperature_K=temperature_K,
            pressure_au=pressure_au,
            tdamp=10 * units.fs,
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
        densities.append(density)
        energies.append(E)
        kinetic_energies.append(Ekin)
        total_energies.append(Etot)
        target_temperatures.append(target_temperature)

        pbar.set_postfix({
            "T(K)": f"{Tnow:.0f}",
            "Ttarget(K)": f"{target_temperature:.0f}",
            "P(GPa)": f"{Pnow:.2f}",
            "rho": f"{density:.3f}",
            "E(eV/a)": f"{E:.4f}",
            "Etot(eV/a)": f"{Etot:.4f}",
        })

    t0 = time.time()
    total_steps = 0
    next_save_step = s

    for start in range(0, rampsteps, updateinterval):
        fraction = start / max(rampsteps - 1, 1)
        target_temperature = T_initial + fraction * (temp_final - T_initial)
        steps_this_chunk = min(updateinterval, rampsteps - start)

        dyn = make_npt_dynamics(target_temperature)
        dyn.run(steps_this_chunk)
        total_steps += steps_this_chunk
        pbar.update(steps_this_chunk)

        if total_steps >= next_save_step:
            write_frame(
                dyn,
                t_fs=(total_steps * MDtimestep) / units.fs,
                target_temperature=target_temperature,
            )
            next_save_step += s

    dyn = make_npt_dynamics(temp_final)

    def update_progress():
        pbar.update(1)

    def write_hold_frame():
        if dyn.get_time() == 0:
            return
        t_fs = ((rampsteps * MDtimestep) + dyn.get_time()) / units.fs
        write_frame(dyn, t_fs=t_fs, target_temperature=temp_final)

    dyn.attach(update_progress, interval=1)
    dyn.attach(write_hold_frame, interval=s)
    dyn.run(holdsteps)

    t1 = time.time()
    pbar.close()
    init_conf.write(checkpoint_fname)

    data = np.column_stack([
        times,
        temperatures,
        pressures,
        densities,
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
            "time_fs temperature_K pressure_GPa density_g_cm3 "
            "energy_eV_per_atom kinetic_energy_eV_per_atom "
            "total_energy_eV_per_atom target_temperature_K"
        ),
    )

    print(f"Temperature ramp finished in {(t1 - t0) / 60:.2f} minutes")
    print(f"Trajectory written to {fname}")
    print(f"Final heated checkpoint written to {checkpoint_fname}")
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

checkpoint_tag = os.path.splitext(os.path.basename(INPUT_CHECKPOINT_XYZ))[0]

run_temperature_ramp(
    init_conf,
    temp_final=T_final,
    pressure_gpa=pressuregpa,
    calc=mace_calc,
    fname=os.path.join(
        MD_RESULTS_DIR,
        f"temperature_ramp_seed_{RUN_SEED}_from_{checkpoint_tag}_P_{pressuregpa:g}GPa_{T_initial:g}K_to_{T_final:g}K.xyz",
    ),
    checkpoint_fname=os.path.join(
        MD_RESULTS_DIR,
        f"heated_checkpoint_seed_{RUN_SEED}_from_{checkpoint_tag}_P_{pressuregpa:g}GPa_T_{T_final:g}K.xyz",
    ),
    s=saveinterval,
)
