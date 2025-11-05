import numpy as np

data = np.load("examples/example_data/ethanol.npz")
R = data["R"]   # positions, shape (frames, atoms, 3)
Z = data["z"]   # atomic numbers

# Optional: map atomic numbers to symbols
from ase.data import chemical_symbols
symbols = [chemical_symbols[int(z)] for z in Z]

with open("ethanol.xyz", "w") as f:
    for frame in R:
        f.write(f"{len(Z)}\n")
        f.write("Frame from NPZ\n")
        for sym, (x, y, z) in zip(symbols, frame):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

print("✅ Saved trajectory as ethanol.xyz")