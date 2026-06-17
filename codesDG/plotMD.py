import numpy as np
import matplotlib.pyplot as plt

# Load thermo data
data = np.load("mace_1500K_density_2.0_thermo.npy")

time_fs = data[:, 0]
temperature = data[:, 1]
pressure = data[:, 2]
energy = data[:, 3]

# Convert fs to ps
time_ps = time_fs / 1000

# Combined plot
fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

axs[0].plot(time_ps, temperature)
axs[0].set_ylabel("T (K)")
axs[0].grid(True)

axs[1].plot(time_ps, pressure)
axs[1].set_ylabel("P (GPa)")
axs[1].grid(True)

axs[2].plot(time_ps, energy)
axs[2].set_ylabel("E (eV/atom)")
axs[2].set_xlabel("Time (ps)")
axs[2].grid(True)

fig.suptitle("MACE MD Thermodynamic Quantities")
plt.tight_layout()
plt.savefig("thermo_summary2.0.png", dpi=300)
plt.show()