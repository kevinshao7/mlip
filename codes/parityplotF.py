from ase.io import read
from mace.calculators import mace_polar
import numpy as np
import matplotlib.pyplot as plt

atoms_list = read(
    "ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz",
    index=":"
)

calc = mace_polar(
    model="checkpoints/polar_ft_1m_run-123.model",       # polar-1-s may not exist; use polar-1-m or polar-1-l
    device="cuda",
    default_dtype="float64"
)

true_F = []
pred_F = []

for i, atoms in enumerate(atoms_list):
    print(i / len(atoms_list))

    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    atoms.calc = calc

    # Ground truth force from your xyz
    true_F.append(atoms.arrays["force"].reshape(-1))

    # Predicted force from MACE
    pred_F.append(atoms.get_forces().reshape(-1))

true_F = np.concatenate(true_F)
pred_F = np.concatenate(pred_F)

np.save("true_F.npy", true_F)
np.save("pred_F.npy", pred_F)

plt.figure(figsize=(6, 6))

plt.scatter(
    true_F,
    pred_F,
    s=1,
    alpha=0.2
)

mn = min(true_F.min(), pred_F.min())
mx = max(true_F.max(), pred_F.max())

plt.plot([mn, mx], [mn, mx], "k--")

plt.xlabel("Ground Truth Force (eV/Å)")
plt.ylabel("Predicted Force (eV/Å)")
plt.title("MACE-POLAR Force Parity Plot")

plt.tight_layout()
plt.savefig("force_parity.png", dpi=300)
plt.show()