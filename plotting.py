from ase.io import read
from mace.calculators import mace_polar
import numpy as np
import matplotlib.pyplot as plt

true_E = np.load("true_E.npy")
pred_E = np.load("pred_E.npy")
# Make parity plot
plt.figure(figsize=(6,6))

plt.scatter(
    true_E,
    pred_E,
    s=8,
    alpha=0.5
)


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
plt.savefig("e_parity.png")
print(true_E-pred_E)