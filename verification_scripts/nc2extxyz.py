from pathlib import Path
import re
import numpy as np
from netCDF4 import Dataset
from ase import Atoms
from ase.io import write
from ase.data import atomic_numbers


def elem_from_name(s: str) -> str:
    """
    Extract chemical element from labels like 'C1', 'H10', 'N', 'Cl2', etc.
    Rule: take leading 1–2 letters; if two letters, second must be lowercase.
    """
    s = s.strip()
    m = re.match(r"([A-Z][a-z]?)", s)
    if not m:
        raise ValueError(f"Cannot parse element from atNames entry: {s!r}")
    return m.group(1)

def to_Z_from_atNames(atNames):
    syms = []
    for raw in atNames[:]:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        syms.append(elem_from_name(str(raw)))
    return np.array([atomic_numbers[s] for s in syms], dtype=int)

def maybe_convert_energy(ev, units: str | None):
    if not units:
        return ev
    u = units.lower()
    if "hartree" in u or u == "ha":
        return ev * 27.211386245988  # Ha -> eV
    if u in ("ev", "electron_volt", "electronvolt"):
        return ev
    # add more if needed
    return ev

def maybe_convert_forces(f, units: str | None):
    # Expect eV/Å for ASE/ML data. Try simple Hartree/Bohr -> eV/Å if detected.
    if not units:
        return f
    u = units.lower()
    if ("hartree" in u or "ha" == u.strip()) and ("bohr" in u):
        ha2ev = 27.211386245988
        bohr2a = 0.529177210903
        return f * (ha2ev / bohr2a)
    if "ev" in u and ("a" in u or "angstrom" in u):
        return f
    return f

# ---- main conversion -------------------------------------------------------

def convert(nc_path: str, out_xyz: str, subsample: int | None = None):
    ds = Dataset(nc_path, "r")

    # geometry
    R_raw = ds.variables["atXYZ"][:]          # (natoms, 3, nframes)
    R = np.transpose(R_raw, (2, 0, 1))        # -> (nframes, natoms, 3)
    nframes, natoms, _ = R.shape

    # species
    Z = to_Z_from_atNames(ds.variables["atNames"])

    # state selection
    energy = ds.variables["energy"][:]        # (nstates, nframes)
    forces = ds.variables["forces"][:]        # (nstates, natoms, 3, nframes)

    if "astate" in ds.variables:
        astate = np.asarray(ds.variables["astate"][:], dtype=int)  # assume 0-based; if 1-based, subtract 1
        # Heuristic: if max(astate) == nstates, it's 1-based
        if astate.max() == energy.shape[0]:
            astate = astate - 1
    else:
        astate = np.zeros(R.shape[0], dtype=int)  # default to state 0

    # units (best-effort)
    e_units = getattr(ds.variables["energy"], "units", None)
    f_units = getattr(ds.variables["forces"], "units", None)

    # (optional) subsample to keep file size reasonable
    idx = np.arange(nframes)
    if subsample and subsample > 1:
        idx = idx[::subsample]

    images = []
    for t in idx:
        s = int(astate[t])
        E_t = float(energy[s, t])
        F_t = forces[s, :, :, t]             # (natoms, 3)

        E_t = maybe_convert_energy(E_t, e_units)
        F_t = maybe_convert_forces(F_t, f_units)

        at = Atoms(numbers=Z, positions=R[t])  # no cell info present
        at.info["energy"] = E_t
        at.arrays["forces"] = np.asarray(F_t, dtype=float)
        images.append(at)

    write(out_xyz, images)
    print(f"✅ Wrote {len(images)} frame(s), {natoms} atoms → {out_xyz}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python nc2extxyz_butene.py input.nc output.extxyz [subsample]")
        sys.exit(1)
    subs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    convert(sys.argv[1], sys.argv[2], subsample=subs)
