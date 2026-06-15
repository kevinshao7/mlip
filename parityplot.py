from ase.io import read
from mace.calculators import mace_polar
import numpy as np
import matplotlib.pyplot as plt

# Load xyz
atoms_list = read(
    "ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz",
    index=":"
)

# Load foundation model
calc = mace_polar(
    model="checkpoints/polar_ft_1m_run-123.model",
    device="cuda",          # or "cpu"
    default_dtype="float64"
)

true_E = []
pred_E = []

# Evaluate all structures
for i in range(len(atoms_list)):
    print(i/len(atoms_list))
    atoms = atoms_list[i]

    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    atoms.calc = calc

    pred_E.append(atoms.get_potential_energy())
    true_E.append(atoms.info["TotEnergy"])

true_E = np.array(true_E)
pred_E = np.array(pred_E)

# Optional: convert to per-atom energies
natoms = len(atoms_list[0])

true_E /= natoms
pred_E /= natoms

# Make parity plot
plt.figure(figsize=(6,6))

plt.scatter(
    true_E,
    pred_E,
    s=8,
    alpha=0.5
)
np.save("true_E",true_E)
np.save("pred_E",pred_E)

mn = min(true_E.min(), pred_E.min())
mx = max(true_E.max(), pred_E.max())

plt.plot(
    [mn, mx],
    [mn, mx],
    'k--'
)

plt.xlabel("Ground Truth Energy (eV/atom)")
plt.ylabel("Predicted Energy (eV/atom)")

plt.title("MACE-POLAR Energy Parity Plot")

plt.tight_layout()
plt.savefig("")