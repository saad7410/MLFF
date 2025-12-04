#!/usr/bin/env python3
"""
Usage:
    python filteringscript_npz.py path/to/my_dynamic.nc

This will create, in the SAME directory as the .nc file:

    my_dynamic_filtered_state_1.npz
    my_dynamic_filtered_state_2.npz
    my_dynamic_filtered_state_3.npz
    ...

One file per electronic state, containing ONLY frames where that state
was the ACTIVE state (according to `astate`), in NPZ format.

Each NPZ contains:
    R : (n_frames, n_atoms, 3)
    F : (n_frames, n_atoms, 3)
    E : (n_frames,)
    z : (n_frames, n_atoms)
"""

import sys
from pathlib import Path
import re

import xarray as xr
import numpy as np
from ase.data import atomic_numbers


def elem_from_name(s: str) -> str:
    """
    Extract chemical element from labels like 'C1', 'H10', 'N', 'Cl2', etc.
    Rule: leading 1–2 letters, first uppercase, optional second lowercase.
    """
    m = re.match(r"([A-Z][a-z]?)", s.strip())
    if not m:
        raise ValueError(f"Cannot parse element from atom name: {s!r}")
    return m.group(1)


def main(nc_path_str: str) -> None:
    nc_path = Path(nc_path_str)
    if not nc_path.is_file():
        raise FileNotFoundError(f"No such file: {nc_path}")

    # open with xarray
    ds = xr.open_dataset(nc_path.as_posix())

    # expect:
    #   energy: (state, time)
    #   forces: (state, atom, xyz, time)
    #   atXYZ:  (atom, xyz, time)
    #   atNames: (atom,)
    #   astate: (time,)
    energy = ds["energy"]
    forces = ds["forces"]
    atXYZ = ds["atXYZ"]
    atNames = ds["atNames"]
    astate = ds["astate"]

    # infer dimension names
    state_dim, time_dim = energy.dims          # e.g. ('state', 'time')
    atom_dim, xyz_dim, _ = atXYZ.dims          # e.g. ('atom', 'xyz', 'time')

    nstates = energy.sizes[state_dim]
    nframes = energy.sizes[time_dim]
    natoms = atXYZ.sizes[atom_dim]

    # astate might be 0-based or 1-based; fix to 0-based
    astate_vals = astate.values.astype(int)
    if astate_vals.max() == nstates:
        astate_vals = astate_vals - 1

    # atom names → atomic numbers (length natoms)
    Z = []
    for raw in atNames.values:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        Z.append(atomic_numbers[elem_from_name(str(raw))])
    Z = np.array(Z, dtype=int)                 # shape: (natoms,)

    stem = nc_path.stem
    print(f"Found {nstates} electronic states, {nframes} frames, {natoms} atoms.")
    print(f"Dimensions: state={state_dim}, time={time_dim}, atom={atom_dim}, xyz={xyz_dim}")

    for state_idx in range(nstates):
        # frames where THIS state is the active state
        mask = (astate_vals == state_idx)
        frame_idx = np.where(mask)[0]

        if frame_idx.size == 0:
            print(f"State {state_idx} (label {state_idx+1}): no frames → skipping.")
            continue

        # select only those time indices for this state
        atXYZ_sel = atXYZ.isel({time_dim: frame_idx})  # (atom, xyz, n_sel)
        energy_sel = energy.isel(
            {state_dim: state_idx, time_dim: frame_idx}
        )                                               # (n_sel,)
        forces_sel = forces.isel(
            {state_dim: state_idx, time_dim: frame_idx}
        )                                               # (atom, xyz, n_sel)

        # convert to NumPy, putting time axis first
        R_all = np.moveaxis(atXYZ_sel.values, -1, 0)    # (n_sel, atom, xyz)
        E_all = energy_sel.values                       # (n_sel,)
        F_all = np.moveaxis(forces_sel.values, -1, 0)   # (n_sel, atom, xyz)

        n_sel = R_all.shape[0]

        # broadcast Z to (n_sel, natoms) as required by SO3krates ('z' key)
        z_full = np.broadcast_to(Z, (n_sel, natoms))    # (n_sel, natoms)

        out_name = f"{stem}_filtered_state_{state_idx+1}.npz"
        out_path = nc_path.with_name(out_name)

        # Only the keys SO3krates expects
        np.savez_compressed(
            out_path.as_posix(),
            R=R_all,
            F=F_all,
            E=E_all,
            z=z_full,
        )

        print(
            f"State {state_idx} (label {state_idx+1}): "
            f"{n_sel} frames → {out_path.name}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filteringscript_npz.py path/to/my_dynamic.nc")
        sys.exit(1)
    main(sys.argv[1])
