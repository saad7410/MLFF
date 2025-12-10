#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import re
import numpy as np
import xarray as xr
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


class ShnitselPreprocessor:
    """
    Helper class for preprocessing Shnitsel dynamic .nc trajectories.

    Responsibilities:
      - Open a .nc file and expose basic metadata (nstates, nframes, natoms).
      - Convert atom names -> atomic numbers (Z).
      - Fix 'astate' to 0-based indices.
      - Export:
          1) A single NPZ with positions for all frames and exactly one
             energy/force per frame, chosen according to the active state
             indicated by `astate`.
          2) One NPZ per electronic state, each containing only frames where
             that state is active (per-state filtering).
    """

    def __init__(self, nc_path: str | Path):
        self.nc_path = Path(nc_path)
        if not self.nc_path.is_file():
            raise FileNotFoundError(f"No such file: {self.nc_path}")

        # Open the netCDF with xarray
        self.ds: xr.Dataset = xr.open_dataset(self.nc_path.as_posix())

        # Expect the following variables:
        #   energy: (state, time)
        #   forces: (state, atom, xyz, time)
        #   atXYZ:  (atom, xyz, time)
        #   atNames: (atom,)
        #   astate: (time,)
        self.energy = self.ds["energy"]
        self.forces = self.ds["forces"]
        self.atXYZ = self.ds["atXYZ"]
        self.atNames = self.ds["atNames"]
        self.astate = self.ds["astate"]

        # Infer dimension names from the xarray dataset
        self.state_dim, self.time_dim = self.energy.dims        # e.g. ('state', 'time')
        self.atom_dim, self.xyz_dim, _ = self.atXYZ.dims        # e.g. ('atom', 'xyz', 'time')

        self.nstates = self.energy.sizes[self.state_dim]
        self.nframes = self.energy.sizes[self.time_dim]
        self.natoms = self.atXYZ.sizes[self.atom_dim]

        # Fix 'astate' to 0-based indexing
        self.astate_vals = self._to_zero_based_astate()

        # Atom names -> atomic numbers (shape: (natoms,))
        self.Z = self._compute_atomic_numbers()

    def _to_zero_based_astate(self) -> np.ndarray:
        """
        Ensure that astate is 0-based:
        - If max(astate) == nstates, interpret as 1-based and subtract 1.
        - Otherwise, keep as is (already 0-based).
        """
        vals = self.astate.values.astype(int)
        if vals.max() == self.nstates:
            vals = vals - 1
        return vals

    def _compute_atomic_numbers(self) -> np.ndarray:
        """
        Convert atNames to atomic numbers using ASE's periodic table.
        """
        Z_list: list[int] = []
        for raw in self.atNames.values:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode()
            symbol = elem_from_name(str(raw))
            Z_list.append(atomic_numbers[symbol])
        return np.array(Z_list, dtype=int)  # (natoms,)



    def export_npz(
        self,
        out_path: Optional[str | Path] = None,
        *,
        z_key: Literal["Z", "z"] = "Z",
        include_all_states: bool = True,
    ) -> Path:
        """
        Export a single NPZ file for this trajectory.

        Contents:
            R      : (n_frames, n_atoms, 3)
                     Atomic positions for all frames in the NC file.
                     (No frames are dropped.)

            E      : (n_frames,)
                     One energy per frame. For each frame t, the energy
                     is selected from the electronic state indicated by
                     `astate[t]`.

            F      : (n_frames, n_atoms, 3)
                     One force field per frame, again selected according
                     to the active state `astate[t]`.

            Z / z  : (n_atoms,)
                     Atomic numbers for the atoms in this system.

            astate : (n_frames,)
                     Active state index (0-based) used to select E and F.

        If `include_all_states` is True, the NPZ also contains:
            energy_all : (n_states, n_frames)
            forces_all : (n_states, n_atoms, 3, n_frames)
        """

        # Positions: move time axis to the front: (atom, xyz, time) -> (time, atom, xyz)
        R = np.moveaxis(self.atXYZ.values, -1, 0)  # (n_frames, n_atoms, 3)

        # Active state energies/forces for each frame
        time_idx = np.arange(self.nframes)  # 0..n_frames-1

        # For frame t, use state = astate_vals[t]
        E = self.energy.values[self.astate_vals, time_idx]          # (n_frames,)
        F = self.forces.values[self.astate_vals, :, :, time_idx]    # (n_frames, n_atoms, 3)

        # Default output path: same as .nc but with .npz
        if out_path is None:
            out_path = self.nc_path.with_suffix(".npz")
        else:
            out_path = Path(out_path)

        # Build dict for np.savez_compressed
        npz_dict = {
            "R": R,
            "F": F,
            "E": E,
            z_key: self.Z,
            "astate": self.astate_vals,
        }

        if include_all_states:
            npz_dict["energy_all"] = self.energy.values
            npz_dict["forces_all"] = self.forces.values

        np.savez_compressed(out_path.as_posix(), **npz_dict)

        print(f"[export_npz] Saved NPZ → {out_path}")
        print("  Shapes:")
        print(f"    R: {R.shape}")
        print(f"    F: {F.shape}")
        print(f"    E: {E.shape}")
        print(f"    {z_key}: {self.Z.shape}")

        return out_path

    def export_states_separately(
        self,
        out_dir: Optional[str | Path] = None,
        *,
        z_key: Literal["Z", "z"] = "z",
        prefix: Optional[str] = None,
    ) -> list[Path]:
        """
        Export one NPZ per electronic state, containing ONLY frames where that
        state is the active state according to `astate`.

        For each state s, creates a file like:
            <stem>_filtered_state_<s+1>.npz

        Each NPZ contains:
            R: (n_sel, n_atoms, 3)
               Positions for the subset of frames where astate == s.

            F: (n_sel, n_atoms, 3)
               Forces for state s, on those frames.

            E: (n_sel,)
               Energies for state s, on those frames.

            z/Z: (n_sel, n_atoms)
               Atomic numbers broadcast over the selected frames.
        """
        if out_dir is None:
            out_dir = self.nc_path.parent
        out_dir = Path(out_dir)

        if prefix is None:
            stem_prefix = self.nc_path.stem
        else:
            stem_prefix = prefix

        print(
            f"[export_states_separately] Found {self.nstates} states, "
            f"{self.nframes} frames, {self.natoms} atoms."
        )
        print(
            f"  Dimensions: state={self.state_dim}, "
            f"time={self.time_dim}, atom={self.atom_dim}, xyz={self.xyz_dim}"
        )

        written_paths: list[Path] = []

        for state_idx in range(self.nstates):
            # Frames where THIS state is active
            mask = (self.astate_vals == state_idx)
            frame_idx = np.where(mask)[0]

            if frame_idx.size == 0:
                print(
                    f"  State {state_idx} (label {state_idx+1}): "
                    "no frames → skipping."
                )
                continue

            # Select only these time indices
            atXYZ_sel = self.atXYZ.isel({self.time_dim: frame_idx})      # (atom, xyz, n_sel)
            energy_sel = self.energy.isel(
                {self.state_dim: state_idx, self.time_dim: frame_idx}
            )                                                             # (n_sel,)
            forces_sel = self.forces.isel(
                {self.state_dim: state_idx, self.time_dim: frame_idx}
            )                                                             # (atom, xyz, n_sel)

            # Reorder to (n_sel, atom, xyz)
            R_all = np.moveaxis(atXYZ_sel.values, -1, 0)
            F_all = np.moveaxis(forces_sel.values, -1, 0)
            E_all = energy_sel.values

            n_sel = R_all.shape[0]

            # Broadcast Z to (n_sel, n_atoms)
            z_full = np.broadcast_to(self.Z, (n_sel, self.natoms))

            out_name = f"{stem_prefix}_filtered_state_{state_idx+1}.npz"
            out_path = out_dir / out_name

            np.savez_compressed(
                out_path.as_posix(),
                R=R_all,
                F=F_all,
                E=E_all,
                **{z_key: z_full},
            )

            print(
                f"  State {state_idx} (label {state_idx+1}): "
                f"{n_sel} frames → {out_path.name}"
            )
            written_paths.append(out_path)

        return written_paths


# -------------------------------------------------------------------------
# Optional: simple CLI wrapper using this class
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess Shnitsel dynamic .nc trajectories for So3krates."
    )
    parser.add_argument("nc_path", help="Path to the .nc file.")
    parser.add_argument(
        "--mode",
        choices=["active", "per-state"],
        default="active",
        help=(
            "'active': one NPZ with one E/F per frame selected by `astate` "
            "(no frame filtering). "
            "'per-state': one NPZ per state, each containing only frames "
            "where that state is active."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file (for 'active') or directory (for 'per-state'). "
             "Defaults: active → <nc>.npz, per-state → same directory as .nc",
    )
    args = parser.parse_args()

    prep = ShnitselPreprocessor(args.nc_path)

    if args.mode == "active":
        prep.export_npz(out_path=args.out)
    else:
        prep.export_states_separately(out_dir=args.out)
