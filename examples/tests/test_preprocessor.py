from __future__ import annotations

from pathlib import Path
import numpy as np

from examples.preprocessing.schnitsel_preprocessor import ShnitselPreprocessor 


def inspect_active_npz(npz_path: Path) -> None:
    print("\n=== CONVERTED NPZ INSPECTION ===")
    print(f"File: {npz_path}")

    data = np.load(npz_path.as_posix(), allow_pickle=False)
    keys = list(data.keys())
    print(f"Keys: {keys}")

    # Print shapes
    for k in keys:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    # Extract main arrays
    R = data["R"]          # (n_frames, n_atoms, 3)
    F = data["F"]          # (n_frames, n_atoms, 3)
    E = data["E"]          # (n_frames,)
    z_key = "z" if "z" in data else "Z"
    Z = data[z_key]

    n_frames, n_atoms, _ = R.shape
    print(f"\nNumber of frames in npz: {n_frames}")
    print(f"Number of atoms: {n_atoms}")

    # Print a few example frames
    n_show = min(3, n_frames)
    print(f"\nShowing first {n_show} frames from npz:")
    for i in range(n_show):
        print(f"\n--- Frame {i} ---")
        print(f"E[{i}] = {E[i]}")
        print(f"R[{i}, 0:3, :] (first 3 atoms):\n{R[i, 0:3, :]}")
        print(f"F[{i}, 0:3, :] (first 3 atoms):\n{F[i, 0:3, :]}")
    print(f"\nAtomic numbers ({z_key}): {Z}")


def inspect_filtered_state_npz(npz_path: Path, state_index: int) -> None:
    print("\n=== FILTERED-STATE NPZ INSPECTION ===")
    print(f"File: {npz_path}  (state index {state_index})")

    data = np.load(npz_path.as_posix(), allow_pickle=False)
    keys = list(data.keys())
    print(f"Keys: {keys}")

    # Print shapes
    for k in keys:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    R = data["R"]          # (n_sel, n_atoms, 3)
    F = data["F"]          # (n_sel, n_atoms, 3)
    E = data["E"]          # (n_sel,)
    z_key = "z" if "z" in data else "Z"
    z_full = data[z_key]   # (n_sel, n_atoms)

    n_sel, n_atoms, _ = R.shape
    print(f"\nNumber of selected frames for this state: {n_sel}")
    print(f"Number of atoms: {n_atoms}")

    # Show a few example frames (energies + small position/force slice)
    n_show = min(3, n_sel)
    print(f"\nShowing first {n_show} frames for this state:")
    for i in range(n_show):
        print(f"\n--- Frame {i} ---")
        print(f"E[{i}] = {E[i]}")
        print(f"R[{i}, 0:3, :] (first 3 atoms):\n{R[i, 0:3, :]}")
        print(f"F[{i}, 0:3, :] (first 3 atoms):\n{F[i, 0:3, :]}")
    print(f"\nFirst row of {z_key} (atomic numbers): {z_full[0]}")


def main() -> None:
    nc_path = Path("/hades/skhan/repos/MLFF/data/schnitsel/fixed/A03_butene_0p50fs_dynamic.nc")

    # Initialize preprocessor
    prep = ShnitselPreprocessor(nc_path)
    print("=== SHNITSEL DATASET METADATA ===")
    print(f"File: {nc_path}")
    print(f"nstates = {prep.nstates}")  
    print(f"nframes = {prep.nframes}")
    print(f"natoms  = {prep.natoms}")
    print(f"state_dim = {prep.state_dim}, time_dim = {prep.time_dim}, atom_dim = {prep.atom_dim}, xyz_dim = {prep.xyz_dim}")

    # 1) Test conversion to single NPZ
    active_npz_path = prep.export_npz(z_key="z")  
    inspect_active_npz(active_npz_path)

    # 2) Test per-state filtering
    filtered_paths = prep.export_states_separately(z_key="z")
    print(f"\nGenerated {len(filtered_paths)} filtered NPZ files for states.")

    # Inspect each filtered file (or just first few)
    for idx, p in enumerate(filtered_paths):
        # idx is 0-based; file name usually encodes state_idx+1
        inspect_filtered_state_npz(p, state_index=idx)


if __name__ == "__main__":
    main()
