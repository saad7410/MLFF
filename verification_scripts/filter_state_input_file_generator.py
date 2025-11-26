"""
Usage:
    python filteringscript.py path/to/my_dynamic.nc

This will create, in the SAME directory as the .nc file:

    my_dynamic_filtered_state_1.extxyz
    my_dynamic_filtered_state_2.extxyz
    my_dynamic_filtered_state_3.extxyz
    ...

One file per electronic state, containing only frames where that state
was the ACTIVE state (according to `astate`).
"""

import sys
from pathlib import Path
import re

import xarray as xr
import numpy as np
from ase import Atoms
from ase.io import write
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

    # figure out dim names from the dataset (more robust than assuming)
    state_dim = energy.dims[0]
    time_dim = energy.dims[1]
    atom_dim = atXYZ.dims[0]
    xyz_dim = atXYZ.dims[1]

    nstates = energy.sizes[state_dim]
    nframes = energy.sizes[time_dim]
    natoms = atXYZ.sizes[atom_dim]

    # astate might be 0-based or 1-based; make it a plain NumPy array first
    astate_vals = astate.values.astype(int)
    if astate_vals.max() == nstates:
        # likely 1-based (1..nstates) → convert to 0..nstates-1
        astate_vals = astate_vals - 1

    # convert atom names → atomic numbers once
    species = []
    for raw in atNames.values:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        species.append(elem_from_name(str(raw)))
    Z = [atomic_numbers[s] for s in species]

    stem = nc_path.stem  # e.g. "A03_butene_0p50fs_dynamic"

    print(f"Found {nstates} electronic states, {nframes} frames, {natoms} atoms.")
    print(f"Dimensions: state={state_dim}, time={time_dim}, atom={atom_dim}, xyz={xyz_dim}")

    for state_idx in range(nstates):
        # frames where THIS state is the active state
        mask = (astate_vals == state_idx)
        frame_idx = np.where(mask)[0]

        if frame_idx.size == 0:
            print(f"State {state_idx} (label {state_idx+1}): no active frames, skipping.")
            continue

        # use .isel to select only those time indices
        atXYZ_sel = atXYZ.isel({time_dim: frame_idx})                # (atom, xyz, n_sel)
        energy_sel = energy.isel({state_dim: state_idx,
                                  time_dim: frame_idx})              # (n_sel,)
        forces_sel = forces.isel({state_dim: state_idx,
                                  time_dim: frame_idx})              # (atom, xyz, n_sel)

        # convert to NumPy for ASE
        pos_all = np.moveaxis(atXYZ_sel.values, -1, 0)   # (n_sel, atom, xyz)
        E_all = energy_sel.values                        # (n_sel,)
        F_all = np.moveaxis(forces_sel.values, -1, 0)    # (n_sel, atom, xyz)

        images = []
        for pos, E, F in zip(pos_all, E_all, F_all):
            atoms = Atoms(numbers=Z, positions=pos)
            atoms.info["energy"] = float(E)
            atoms.arrays["forces"] = np.asarray(F, dtype=float)
            images.append(atoms)

        out_name = f"{stem}_filtered_state_{state_idx+1}.extxyz"
        out_path = nc_path.with_name(out_name)
        write(out_path.as_posix(), images)
        print(f"State {state_idx} (label {state_idx+1}): "
              f"{len(images)} frames → {out_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filteringscript.py path/to/my_dynamic.nc")
        sys.exit(1)
    main(sys.argv[1])
