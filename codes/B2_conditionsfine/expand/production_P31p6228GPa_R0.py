import os
import shutil

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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputsfull", "conditionsfine", "P31p6228GPa_R0")
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "outputsfull", ".cache"))

PACKMOL_EXE = shutil.which("packmol")
if PACKMOL_EXE is None:
    raise RuntimeError(
        "packmol was not found on PATH. SimCell.build() needs Packmol for "
        "well-behaved initial molecular packing."
    )
print(f"Using Packmol executable: {PACKMOL_EXE}")

# Generate initial configuration.

densitygcm3 = 0.2 # initial build density, g/cm3
target_profile_densitygcm3 = 2.032 # log-pressure-interpolated profile density, g/cm3
pressuregpa = 31.6227766017 # GPa
targetmolecules = 100
moleculemass = 18.01528 # grams per mol, composition-weighted
pressure_start_gpa = 0.0001 # GPa
pressureramptime = 200*1000*units.fs
tempramptime = 200*1000*units.fs
MDtimestep = 0.5*units.fs
totaltimesteps = 8000000  # 4 ns at 0.5 fs/step
saveinterval =5
T_initial=300 #always keep at room temperature
T_final = 2380  # log-pressure-interpolated preferred_uranus_temperature_K
composition_label = "R0"
ammonia_water_ratio = 0 # NH3/H2O molar ratio

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

simbox.add_solvent([water], ratio=[1], zdim=boxsize, density=densitygcm3)
print("Building initial molecular coordinates with SimCell.build()/Packmol...")
simbox.build(padding=0.5)

atoms = simbox.to_ase()    
init_conf =atoms
print("Number of atoms:", len(init_conf))
print("Chemical formula:", init_conf.get_chemical_formula())
print("Initial build density g/cm3:", densitygcm3)
print("Uranus profile density g/cm3:", target_profile_densitygcm3)

init_conf.info["charge"] = 0
init_conf.info["spin"] = 1
init_conf.info["external_field"] = [0.0, 0.0, 0.0]

def stagedMD(init_conf, temp_final, pressure_gpa, calc, fname, checkpoint_fname, s, T, T_thermo=100):
    # s is save interval, T is total NPT integration steps after initial NVT.
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
    pressure_ramp_steps = int(pressureramptime / MDtimestep)
    temperature_ramp_steps = int(tempramptime / MDtimestep)
    hold_steps = T - pressure_ramp_steps - temperature_ramp_steps
    if hold_steps < 0:
        raise ValueError(
            "totaltimesteps must be at least pressure_ramp_steps + "
            "temperature_ramp_steps"
        )

    print(f"pressure_ramp_steps {pressure_ramp_steps}")
    print(f"temperature_ramp_steps {temperature_ramp_steps}")
    print(f"hold_steps {hold_steps}")

    output_dir = os.path.dirname(fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(fname):
        os.remove(fname)

    # arrays for plotting all staged NPT legs.
    times = []
    stages = []
    temperatures = []
    pressures = []
    densities = []
    energies = []
    kinetic_energies = []
    total_energies = []
    target_temperatures = []
    target_pressures = []

    pbar = tqdm(
        total=T,
        desc=(
            f"NPT staged equilibration: pressure {pressure_start_gpa:g} -> "
            f"{pressure_gpa:g} GPa at {T_initial:g} K, then temperature "
            f"{T_initial:g} -> {temp_final:g} K"
        ),
    )

    def make_npt_dynamics(temperature_K, pressure_target_gpa):
        return IsotropicMTKNPT(
            init_conf,
            timestep=MDtimestep,
            temperature_K=temperature_K,
            pressure_au=pressure_target_gpa * units.GPa,
            tdamp=100 * units.fs,
            pdamp=1000 * units.fs
        )

    def write_frame(dyn, t_fs, stage, target_temperature, target_pressure):
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
        stages.append(stage)
        temperatures.append(Tnow)
        pressures.append(Pnow)
        densities.append(density)
        energies.append(E)
        kinetic_energies.append(Ekin)
        total_energies.append(Etot)
        target_temperatures.append(target_temperature)
        target_pressures.append(target_pressure)

        pbar.set_postfix({
            "stage": stage,
            "T(K)": f"{Tnow:.0f}",
            "Ttarget(K)": f"{target_temperature:.0f}",
            "P(GPa)": f"{Pnow:.2f}",
            "Ptarget(GPa)": f"{target_pressure:.2f}",
            "rho": f"{density:.3f}",
            "E(eV/a)": f"{E:.4f}",
            "Etot(eV/a)": f"{Etot:.4f}",
        })

    t0 = time.time()
    total_steps = 0
    next_save_step = s
    updateinterval = max(1, min(20, s))

    def run_chunked_stage(stage, steps, temperature_start, temperature_end, pressure_start, pressure_end):
        nonlocal total_steps, next_save_step
        if steps <= 0:
            return

        for start in range(0, steps, updateinterval):
            fraction = start / max(steps - 1, 1)
            target_temperature = temperature_start + fraction * (temperature_end - temperature_start)
            target_pressure = pressure_start + fraction * (pressure_end - pressure_start)
            steps_this_chunk = min(updateinterval, steps - start)

            dyn = make_npt_dynamics(target_temperature, target_pressure)
            dyn.run(steps_this_chunk)
            total_steps += steps_this_chunk
            pbar.update(steps_this_chunk)

            if total_steps >= next_save_step:
                write_frame(
                    dyn,
                    t_fs=(total_steps * MDtimestep) / units.fs,
                    stage=stage,
                    target_temperature=target_temperature,
                    target_pressure=target_pressure,
                )
                next_save_step += s

    run_chunked_stage(
        stage="pressure_ramp",
        steps=pressure_ramp_steps,
        temperature_start=T_initial,
        temperature_end=T_initial,
        pressure_start=pressure_start_gpa,
        pressure_end=pressure_gpa,
    )
    run_chunked_stage(
        stage="temperature_ramp",
        steps=temperature_ramp_steps,
        temperature_start=T_initial,
        temperature_end=temp_final,
        pressure_start=pressure_gpa,
        pressure_end=pressure_gpa,
    )
    run_chunked_stage(
        stage="production_hold",
        steps=hold_steps,
        temperature_start=temp_final,
        temperature_end=temp_final,
        pressure_start=pressure_gpa,
        pressure_end=pressure_gpa,
    )
    t1 = time.time()

    pbar.close()
    dyn.atoms.write(checkpoint_fname)

    data = np.column_stack([
        times,
        temperatures,
        pressures,
        densities,
        energies,
        kinetic_energies,
        total_energies,
        target_temperatures,
        target_pressures,
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
            "total_energy_eV_per_atom target_temperature_K target_pressure_GPa"
        ),
    )

    stage_name = fname.replace(".xyz", "_stages.txt")
    with open(stage_name, "w", encoding="utf-8") as handle:
        handle.write("time_fs stage\n")
        for t_fs, stage in zip(times, stages):
            handle.write(f"{t_fs:.12g} {stage}\n")

    print(f"MD finished in {(t1 - t0) / 60:.2f} minutes")
    print(f"Trajectory written to {fname}")
    print(f"Final checkpoint written to {checkpoint_fname}")
    print(f"Thermo data written to {npyname}")
    print(f"Text data written to {txtname}")
    print(f"Stage labels written to {stage_name}")




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

stagedMD(
    init_conf,
    temp_final=T_final,
    pressure_gpa=pressuregpa,
    calc=mace_calc,
    fname=os.path.join(
        MD_RESULTS_DIR,
        f"production_seed_{RUN_SEED}_P_{pressuregpa:g}GPa_T_{T_final:g}K_{composition_label}_initial_density_{densitygcm3}_profile_density_{target_profile_densitygcm3}.xyz",
    ),
    checkpoint_fname=os.path.join(
        MD_RESULTS_DIR,
        f"checkpoint_seed_{RUN_SEED}_P_{pressuregpa:g}GPa_T_{T_final:g}K_{composition_label}_initial_density_{densitygcm3}_profile_density_{target_profile_densitygcm3}.xyz",
    ),
    s=5,
    T=totaltimesteps
)
