#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import re
from ase.data import atomic_numbers


def elem_from_name(s: str) -> str:
    m = re.match(r"([A-Z][a-z]?)", s.strip())
    if not m:
        raise ValueError(f"Cannot parse element: {s}")
    return m.group(1)


def main(nc_path_str):
    nc_path = Path(nc_path_str)
    ds = xr.open_dataset(nc_path.as_posix())

    # Load arrays
    atXYZ = ds["atXYZ"]           # (atom, xyz, time)
    forces = ds["forces"]         # (state, atom, xyz, time)
    energy = ds["energy"]         # (state, time)
    astate = ds["astate"]         # (time,)
    atNames = ds["atNames"]       # (atom,)

    nstates = energy.sizes[energy.dims[0]]
    natoms = atXYZ.sizes[atXYZ.dims[0]]
    nframes = atXYZ.sizes[atXYZ.dims[2]]

    # Fix astate to 0-based
    astate_np = astate.values.astype(int)
    if astate_np.max() == nstates:
        astate_np = astate_np - 1

    # Convert names -> atomic numbers
    Z = []
    for raw in atNames.values:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        Z.append(atomic_numbers[elem_from_name(str(raw))])
    Z = np.array(Z, dtype=int)

    # Extract positions: move time last -> time first
    R = np.moveaxis(atXYZ.values, -1, 0)  # (time, atom, xyz)

    # Extract energies/forces on ACTIVE state
    E = energy.values[astate_np, np.arange(nframes)]           # (time,)
    F = forces.values[astate_np, :, :, np.arange(nframes)]     # (time, atom, xyz)

    out_name = nc_path.with_suffix(".npz")

    np.savez_compressed(
        out_name.as_posix(),
        R=R,
        F=F,
        E=E,
        Z=Z,
        astate=astate_np,
        energy_all=energy.values,
        forces_all=forces.values,
    )

    print(f"Saved NPZ → {out_name}")
    print(f"Shapes:")
    print(f"  R: {R.shape}")
    print(f"  F: {F.shape}")
    print(f"  E: {E.shape}")
    print(f"  Z: {Z.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python nc_to_npz.py path/to/file.nc")
        sys.exit(1)

    main(sys.argv[1])
