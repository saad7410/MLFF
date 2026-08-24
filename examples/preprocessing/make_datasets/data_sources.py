from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from netCDF4 import Dataset

try:
    from .preprocessing_helpers import (
        StateData,
        discover_molecule_ids,
        load_filtered_state,
        validate_state_arrays,
    )
except ImportError:  # Support importing this module from a directly run script.
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from examples.preprocessing.make_datasets.preprocessing_helpers import (  # type: ignore[no-redef]
        StateData,
        discover_molecule_ids,
        load_filtered_state,
        validate_state_arrays,
    )


BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.4220674763

_OUTPUT_UNITS = {
    "R": "angstrom",
    "E": "eV",
    "F": "eV/angstrom",
}

_ELEMENT_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
_ATOMIC_NUMBERS = {symbol: index for index, symbol in enumerate(_ELEMENT_SYMBOLS, 1)}


@runtime_checkable
class StateSource(Protocol):
    """Source-neutral interface consumed by the dataset builders."""

    def discover_molecule_ids(self, requested: list[str] | None = None) -> list[str]:
        ...

    def load_state(
        self,
        mol_id: str,
        state_label: int,
        *,
        active_only: bool = False,
    ) -> StateData:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class NpyDirectorySource:
    """Adapter for the existing molecule/state ``*.npy`` directory layout."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def discover_molecule_ids(self, requested: list[str] | None = None) -> list[str]:
        return discover_molecule_ids(self.root, requested)

    def load_state(
        self,
        mol_id: str,
        state_label: int,
        *,
        active_only: bool = False,
    ) -> StateData:
        # Each state directory is already a state-specific filtered source, so
        # active_only does not require a second selection step here.
        del active_only
        return load_filtered_state(self.root, mol_id, state_label)

    def metadata(self) -> dict[str, Any]:
        return {
            "source_type": "npy_directory",
            "source_root": self.root.resolve().as_posix(),
            "output_units": dict(_OUTPUT_UNITS),
            "conversion_factors": {"R": 1.0, "E": 1.0, "F": 1.0},
        }


class NpzFileSource:
    """Read aggregate or filtered molecule/state arrays from ``.npz`` files.

    Aggregate archives contain ``R``, ``energy_all``, ``forces_all``, ``astate``,
    and ``z``. Filtered archives contain ``R``, ``F``, ``E``, and ``z`` and use
    one-based filename tags such as ``state_2``. Molecule tags are inferred from
    every filename, and input order determines molecule order.
    """

    def __init__(self, sources: Sequence[str | Path]):
        if isinstance(sources, (str, Path)):
            entries = [sources]
        else:
            entries = list(sources)
        (
            self._paths,
            self._aggregate_paths,
            self._molecule_ids,
        ) = _validate_npz_sources(entries)
        self._cache: dict[tuple[str, int, bool], StateData] = {}

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> NpzFileSource:
        return cls(paths)

    @property
    def molecule_ids(self) -> list[str]:
        return list(self._molecule_ids)

    def discover_molecule_ids(self, requested: list[str] | None = None) -> list[str]:
        available = list(self._molecule_ids)
        if requested is None:
            return available

        canonical_by_casefold = {mol_id.casefold(): mol_id for mol_id in available}
        selected: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for requested_id in requested:
            key = str(requested_id).casefold()
            canonical = canonical_by_casefold.get(key)
            if canonical is None:
                missing.append(str(requested_id))
                continue
            if key in seen:
                raise ValueError(f"Molecule IDs must be unique, got {requested}.")
            selected.append(canonical)
            seen.add(key)
        if missing:
            raise FileNotFoundError(
                f"Requested molecules do not have NPZ sources: {missing}. "
                f"Available molecules: {available}."
            )
        return selected

    def load_state(
        self,
        mol_id: str,
        state_label: int,
        *,
        active_only: bool = False,
    ) -> StateData:
        canonical_id = self._canonical_molecule_id(mol_id)
        key = (canonical_id, int(state_label))
        path = self._paths.get(key)
        if path is None:
            available_states = sorted(
                state for source_id, state in self._paths if source_id == canonical_id
            )
            raise FileNotFoundError(
                f"No NPZ source for {canonical_id} state {int(state_label)}; "
                f"available zero-based states are {available_states}."
            )

        is_aggregate = path in self._aggregate_paths
        # A filtered state archive already represents active-only rows.
        effective_active_only = bool(active_only and is_aggregate)
        cache_key = (canonical_id, int(state_label), effective_active_only)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        state = _read_npz_state(
            canonical_id,
            int(state_label),
            path,
            aggregate=is_aggregate,
            active_only=effective_active_only,
        )
        self._cache[cache_key] = state
        return state

    def metadata(self) -> dict[str, Any]:
        return {
            "source_type": "npz_files",
            "source_files": [
                {
                    "mol_id": mol_id,
                    "states": sorted(
                        state
                        for (source_id, state), source_path in self._paths.items()
                        if source_id == mol_id and source_path == path
                    ),
                    "path": path.as_posix(),
                    "layout": (
                        "aggregate_all_states"
                        if path in self._aggregate_paths
                        else "filtered_state"
                    ),
                }
                for mol_id, path in dict.fromkeys(
                    (source_id, source_path)
                    for (source_id, _), source_path in self._paths.items()
                )
            ],
            "unit_policy": "preserve input values without unit conversion",
            "units": {"R": "as provided", "E": "as provided", "F": "as provided"},
            "conversion_factors": {"R": 1.0, "E": 1.0, "F": 1.0},
        }

    def _canonical_molecule_id(self, mol_id: str) -> str:
        requested = str(mol_id)
        matches = [
            available
            for available in self._molecule_ids
            if available.casefold() == requested.casefold()
        ]
        if not matches:
            raise FileNotFoundError(
                f"No NPZ source registered for molecule {requested!r}; "
                f"available molecules are {list(self._molecule_ids)}."
            )
        return matches[0]


@dataclass(frozen=True)
class _NetCDFRecord:
    mol_id: str
    path: Path
    R: np.ndarray
    F: np.ndarray
    E: np.ndarray
    z: np.ndarray
    active_state: np.ndarray
    state_axis_by_label: dict[int, int]
    metadata: dict[str, Any]


class NetCDFSource:
    """Read and normalize one NetCDF trajectory stack per molecule.

    ``sources`` may be a mapping, a sequence of ``(molecule_id, path)`` pairs,
    or the same ``PATH``/``MOL_ID=PATH`` strings accepted by :meth:`from_specs`.
    Input order is retained as molecule order.
    """

    def __init__(
        self,
        sources: (
            Mapping[str, str | Path]
            | Sequence[str | Path]
            | Sequence[tuple[str, str | Path]]
        ),
    ):
        parsed = _coerce_sources(sources)
        self._paths = _validate_sources(parsed)
        self._cache: dict[str, _NetCDFRecord] = {}

    @classmethod
    def from_specs(cls, specs: Sequence[str | Path]) -> NetCDFSource:
        return cls(specs)

    @property
    def molecule_ids(self) -> list[str]:
        return list(self._paths)

    def discover_molecule_ids(self, requested: list[str] | None = None) -> list[str]:
        available = list(self._paths)
        if requested is None:
            return available

        canonical_by_casefold = {mol_id.casefold(): mol_id for mol_id in available}
        selected: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for requested_id in requested:
            key = str(requested_id).casefold()
            canonical = canonical_by_casefold.get(key)
            if canonical is None:
                missing.append(str(requested_id))
                continue
            if key in seen:
                raise ValueError(f"Molecule IDs must be unique, got {requested}.")
            selected.append(canonical)
            seen.add(key)
        if missing:
            raise FileNotFoundError(
                f"Requested molecules do not have NetCDF sources: {missing}. "
                f"Available molecules: {available}."
            )
        return selected

    def load_state(
        self,
        mol_id: str,
        state_label: int,
        *,
        active_only: bool = False,
    ) -> StateData:
        canonical_id = self._canonical_molecule_id(mol_id)
        record = self._load_record(canonical_id)
        normalized_state = int(state_label)
        if normalized_state not in record.state_axis_by_label:
            available = sorted(record.state_axis_by_label)
            raise ValueError(
                f"{canonical_id}: state {normalized_state} is unavailable; "
                f"normalized states are {available}."
            )

        if active_only:
            frame_idx = np.flatnonzero(record.active_state == normalized_state).astype(
                np.int64
            )
        else:
            frame_idx = np.arange(record.R.shape[0], dtype=np.int64)

        state_axis = record.state_axis_by_label[normalized_state]
        R = np.ascontiguousarray(record.R[frame_idx], dtype=np.float32)
        F = np.ascontiguousarray(record.F[state_axis, frame_idx], dtype=np.float32)
        E = np.ascontiguousarray(record.E[state_axis, frame_idx], dtype=np.float32)
        z_per_frame = np.broadcast_to(record.z, (len(frame_idx), len(record.z))).copy()
        node_mask = np.ones(z_per_frame.shape, dtype=bool)

        return StateData(
            mol_id=canonical_id,
            state_label=normalized_state,
            path=record.path,
            R=R,
            F=F,
            E=E,
            z_per_frame=z_per_frame,
            node_mask=node_mask,
            frame_idx=frame_idx,
        )

    def metadata(self) -> dict[str, Any]:
        records = [self._load_record(mol_id) for mol_id in self._paths]
        return {
            "source_type": "netcdf",
            "source_files": [
                {"mol_id": mol_id, "path": path.as_posix()}
                for mol_id, path in self._paths.items()
            ],
            "output_units": dict(_OUTPUT_UNITS),
            "source_molecule_data": [record.metadata for record in records],
        }

    def _canonical_molecule_id(self, mol_id: str) -> str:
        requested = str(mol_id)
        matches = [
            available
            for available in self._paths
            if available.casefold() == requested.casefold()
        ]
        if not matches:
            raise FileNotFoundError(
                f"No NetCDF source registered for molecule {requested!r}; "
                f"available molecules are {list(self._paths)}."
            )
        return matches[0]

    def _load_record(self, mol_id: str) -> _NetCDFRecord:
        cached = self._cache.get(mol_id)
        if cached is not None:
            return cached
        record = _read_netcdf_record(mol_id, self._paths[mol_id])
        self._cache[mol_id] = record
        return record


def _coerce_sources(
    sources: (
        Mapping[str, str | Path]
        | Sequence[str | Path]
        | Sequence[tuple[str, str | Path]]
    ),
) -> list[tuple[str, Path]]:
    if isinstance(sources, Mapping):
        return [(_validate_explicit_id(mol_id), Path(path)) for mol_id, path in sources.items()]

    if isinstance(sources, (str, Path)):
        entries: list[Any] = [sources]
    else:
        entries = list(sources)
    if not entries:
        raise ValueError("At least one NetCDF source is required.")

    if all(isinstance(entry, tuple) and len(entry) == 2 for entry in entries):
        return [
            (_validate_explicit_id(entry[0]), Path(entry[1]))
            for entry in entries
        ]
    if any(isinstance(entry, tuple) for entry in entries):
        raise TypeError(
            "NetCDF sources must be all PATH/MOL_ID=PATH specs or all "
            "(MOL_ID, PATH) pairs."
        )
    return [_parse_source_spec(entry) for entry in entries]


def _parse_source_spec(spec: str | Path) -> tuple[str, Path]:
    if isinstance(spec, Path):
        path = spec
        return _infer_molecule_id(path), path

    text = str(spec).strip()
    if not text:
        raise ValueError("NetCDF input specs must not be empty.")

    # Existing filenames containing '=' remain usable as plain paths. For a
    # nonexistent path, '=' denotes the explicit MOL_ID=PATH form.
    plain_path = Path(text)
    if "=" not in text or plain_path.is_file():
        return _infer_molecule_id(plain_path), plain_path

    mol_id_text, path_text = text.split("=", 1)
    if not path_text.strip():
        raise ValueError(f"NetCDF input spec has no path: {spec!r}.")
    return _validate_explicit_id(mol_id_text), Path(path_text.strip())


def _validate_explicit_id(mol_id: Any) -> str:
    normalized = str(mol_id).strip()
    if not normalized:
        raise ValueError("An explicit NetCDF molecule ID must not be empty.")
    if any(separator in normalized for separator in ("/", "\\", "=")):
        raise ValueError(
            f"Invalid molecule ID {normalized!r}; IDs cannot contain path separators or '='."
        )
    return normalized


def _infer_molecule_id(path: Path) -> str:
    candidates = {
        match.group(1).upper()
        for match in re.finditer(
            r"(?i)(?<![a-z0-9])([a-z]+[0-9]+)(?![a-z0-9])",
            path.stem,
        )
    }
    if not candidates:
        raise ValueError(
            f"Cannot infer a molecule ID from filename {path.name!r}; "
            "the name must contain one letter-and-number tag such as A03."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous molecule IDs in filename {path.name!r}: "
            f"{sorted(candidates)}. The name must contain exactly one molecule tag."
        )
    return candidates.pop()


def _infer_npz_state_label(path: Path) -> int:
    matches = {
        int(match.group(1))
        for match in re.finditer(
            r"(?i)(?<![a-z0-9])state[_-]?([0-9]+)(?![a-z0-9])",
            path.stem,
        )
    }
    if not matches:
        raise ValueError(
            f"Cannot infer a state from NPZ filename {path.name!r}; "
            "the name must contain a one-based state tag such as state_1."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous state tags in NPZ filename {path.name!r}: "
            f"{sorted(matches)}. The name must contain exactly one state tag."
        )
    source_state = matches.pop()
    if source_state < 1:
        raise ValueError(
            f"NPZ filename {path.name!r} has state_{source_state}; "
            "filename state tags must be one-based."
        )
    return source_state - 1


def _validate_npz_sources(
    entries: Sequence[str | Path],
) -> tuple[dict[tuple[str, int], Path], set[Path], list[str]]:
    if not entries:
        raise ValueError("At least one NPZ source is required.")

    result: dict[tuple[str, int], Path] = {}
    aggregate_paths: set[Path] = set()
    molecule_ids: list[str] = []
    seen_molecules: set[str] = set()
    seen_paths: set[Path] = set()
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No such NPZ source: {path}")
        if path.suffix.casefold() != ".npz":
            raise ValueError(f"NPZ source must have a .npz suffix, got: {path}")
        if path in seen_paths:
            raise ValueError(f"NPZ source was provided more than once: {path}")
        seen_paths.add(path)

        mol_id = _infer_molecule_id(path)
        state_labels, is_aggregate = _inspect_npz_layout(path)
        if is_aggregate:
            aggregate_paths.add(path)
        for state_label in state_labels:
            key = (mol_id, state_label)
            if key in result:
                raise ValueError(
                    f"Duplicate NPZ source for molecule/state {key}: "
                    f"{result[key]} and {path}."
                )
            result[key] = path

        mol_key = mol_id.casefold()
        if mol_key not in seen_molecules:
            molecule_ids.append(mol_id)
            seen_molecules.add(mol_key)
    return result, aggregate_paths, molecule_ids


def _inspect_npz_layout(path: Path) -> tuple[list[int], bool]:
    with np.load(path, allow_pickle=False) as archive:
        aggregate_keys = {"R", "energy_all", "forces_all", "astate", "z"}
        if aggregate_keys <= set(archive.files):
            R_shape = archive["R"].shape
            if len(R_shape) != 3 or R_shape[-1] != 3:
                raise ValueError(
                    f"{path}: aggregate R must be [T,N,3], got {R_shape}."
                )
            nframes, natoms, _ = R_shape
            energies = _normalize_npz_energies(
                archive["energy_all"], nframes=nframes, path=path
            )
            _normalize_npz_forces(
                archive["forces_all"],
                nstates=energies.shape[0],
                nframes=nframes,
                natoms=natoms,
                path=path,
            )
            return list(range(energies.shape[0])), True

        required = {"R", "F", "E", "z"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"{path}: expected an aggregate all-state archive or a filtered "
                f"state archive; missing required arrays {missing}."
            )
    return [_infer_npz_state_label(path)], False


def _normalize_npz_energies(
    raw: np.ndarray,
    *,
    nframes: int,
    path: Path,
) -> np.ndarray:
    energies = np.asarray(raw)
    if energies.ndim != 2:
        raise ValueError(
            f"{path}: energy_all must be two-dimensional, got {energies.shape}."
        )
    if energies.shape[1] == nframes:
        return energies
    if energies.shape[0] == nframes:
        return energies.T
    raise ValueError(
        f"{path}: energy_all shape {energies.shape} has no frame axis of "
        f"length {nframes}."
    )


def _normalize_npz_forces(
    raw: np.ndarray,
    *,
    nstates: int,
    nframes: int,
    natoms: int,
    path: Path,
) -> np.ndarray:
    forces = np.asarray(raw)
    target_shape = (nstates, nframes, natoms, 3)
    if forces.shape == target_shape:
        return forces
    if forces.shape == (nstates, natoms, 3, nframes):
        return forces.transpose(0, 3, 1, 2)
    if forces.shape == (nframes, nstates, natoms, 3):
        return forces.transpose(1, 0, 2, 3)
    raise ValueError(
        f"{path}: forces_all shape {forces.shape} cannot be normalized to "
        f"{target_shape}."
    )


def _read_npz_state(
    mol_id: str,
    state_label: int,
    path: Path,
    *,
    aggregate: bool,
    active_only: bool,
) -> StateData:
    with np.load(path, allow_pickle=False) as archive:
        R_raw = np.asarray(archive["R"])
        nframes, natoms, _ = R_raw.shape
        if aggregate:
            E_all = _normalize_npz_energies(
                archive["energy_all"], nframes=nframes, path=path
            )
            F_all = _normalize_npz_forces(
                archive["forces_all"],
                nstates=E_all.shape[0],
                nframes=nframes,
                natoms=natoms,
                path=path,
            )
            active_state, _ = _normalize_active_state(
                np.asarray(archive["astate"]),
                nstates=E_all.shape[0],
                preferred_base=0,
                path=path,
            )
            if active_only:
                selected = np.flatnonzero(active_state == state_label).astype(np.int64)
            else:
                selected = np.arange(nframes, dtype=np.int64)
            E_raw = E_all[state_label, selected]
            F_raw = F_all[state_label, selected]
            R_raw = R_raw[selected]
        else:
            E_raw = np.asarray(archive["E"]).reshape(-1)
            F_raw = np.asarray(archive["F"])
            selected = np.arange(E_raw.shape[0], dtype=np.int64)

        R = _convert_to_float32(R_raw, 1.0, path=path, variable_name="R")
        F = _convert_to_float32(
            F_raw,
            1.0,
            path=path,
            variable_name="F" if not aggregate else "forces_all",
        )
        E = _convert_to_float32(
            E_raw,
            1.0,
            path=path,
            variable_name="E" if not aggregate else "energy_all",
        ).reshape(-1)

        z_raw = np.asarray(archive["z"], dtype=np.int64)
        if z_raw.ndim == 1:
            z_per_frame = np.broadcast_to(
                z_raw, (selected.shape[0], z_raw.shape[0])
            ).copy()
        elif z_raw.ndim == 2:
            z_per_frame = z_raw[selected]
        else:
            raise ValueError(f"{path}: z must be [N] or [T,N], got {z_raw.shape}.")
        if "node_mask" in archive:
            raw_node_mask = np.asarray(archive["node_mask"], dtype=bool)
            node_mask = raw_node_mask[selected]
        else:
            node_mask = z_per_frame > 0
        if "frame_idx" in archive:
            raw_frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64)
            frame_idx = raw_frame_idx[selected]
        else:
            frame_idx = selected

    arrays = {
        "R": R,
        "F": F,
        "E": E,
        "z_per_frame": z_per_frame,
        "node_mask": node_mask,
        "frame_idx": frame_idx,
    }
    validate_state_arrays(mol_id=mol_id, state_label=state_label, arrays=arrays)
    return StateData(
        mol_id=mol_id,
        state_label=state_label,
        path=path,
        R=R,
        F=F,
        E=E,
        z_per_frame=z_per_frame,
        node_mask=node_mask,
        frame_idx=frame_idx,
    )


def _validate_sources(entries: list[tuple[str, Path]]) -> dict[str, Path]:
    if not entries:
        raise ValueError("At least one NetCDF source is required.")

    result: dict[str, Path] = {}
    ids_by_casefold: dict[str, str] = {}
    ids_by_path: dict[Path, str] = {}
    for mol_id, unvalidated_path in entries:
        key = mol_id.casefold()
        if key in ids_by_casefold:
            raise ValueError(
                f"Duplicate NetCDF molecule ID {mol_id!r}; it conflicts with "
                f"{ids_by_casefold[key]!r}."
            )

        path = unvalidated_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No such NetCDF source for {mol_id}: {path}")
        if path.suffix.casefold() != ".nc":
            raise ValueError(f"NetCDF source must have a .nc suffix, got: {path}")
        if path in ids_by_path:
            raise ValueError(
                f"NetCDF source {path} was assigned to both {ids_by_path[path]!r} "
                f"and {mol_id!r}."
            )

        result[mol_id] = path
        ids_by_casefold[key] = mol_id
        ids_by_path[path] = mol_id
    return result


def _read_netcdf_record(mol_id: str, path: Path) -> _NetCDFRecord:
    with Dataset(path.as_posix(), mode="r") as dataset:
        required = ("energy", "forces", "atXYZ", "astate")
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ValueError(f"{path}: missing required root variables {missing}.")

        energy_var = dataset.variables["energy"]
        forces_var = dataset.variables["forces"]
        positions_var = dataset.variables["atXYZ"]
        active_var = dataset.variables["astate"]

        E_raw = _read_axis_normalized(
            energy_var, ("state", "frame"), path=path, variable_name="energy"
        )
        F_raw = _read_axis_normalized(
            forces_var,
            ("state", "frame", "atom", "direction"),
            path=path,
            variable_name="forces",
        )
        R_raw = _read_axis_normalized(
            positions_var,
            ("frame", "atom", "direction"),
            path=path,
            variable_name="atXYZ",
        )
        active_raw = _read_axis_normalized(
            active_var, ("frame",), path=path, variable_name="astate"
        )

        nstates, nframes = E_raw.shape
        if F_raw.shape != (nstates, nframes, R_raw.shape[1], 3):
            raise ValueError(
                f"{path}: normalized forces shape {F_raw.shape} does not match "
                f"state/frame/atom dimensions {(nstates, nframes, R_raw.shape[1], 3)}."
            )
        if R_raw.shape != (nframes, F_raw.shape[2], 3):
            raise ValueError(
                f"{path}: normalized atXYZ shape {R_raw.shape} is incompatible with "
                f"energy {E_raw.shape} and forces {F_raw.shape}."
            )
        if active_raw.shape != (nframes,):
            raise ValueError(
                f"{path}: normalized astate must have shape {(nframes,)}, "
                f"got {active_raw.shape}."
            )
        if nstates == 0 or nframes == 0 or R_raw.shape[1] == 0:
            raise ValueError(
                f"{path}: state, frame, and atom dimensions must be non-empty; "
                f"got states={nstates}, frames={nframes}, atoms={R_raw.shape[1]}."
            )

        z, z_source = _read_atomic_numbers(dataset, path, R_raw.shape[1])
        state_raw, state_base = _read_state_coordinate(dataset, path, nstates)
        state_axis_by_label = {
            int(source_label - state_base): axis
            for axis, source_label in enumerate(state_raw)
        }
        active_state, active_base = _normalize_active_state(
            active_raw,
            nstates=nstates,
            preferred_base=state_base,
            path=path,
        )

        position_unit, position_factor = _unit_conversion(
            positions_var, quantity="R", path=path
        )
        energy_unit, energy_factor = _unit_conversion(
            energy_var, quantity="E", path=path
        )
        force_unit, force_factor = _unit_conversion(
            forces_var, quantity="F", path=path
        )

        R = _convert_to_float32(
            R_raw, position_factor, path=path, variable_name="atXYZ"
        )
        E = _convert_to_float32(
            E_raw, energy_factor, path=path, variable_name="energy"
        )
        F = _convert_to_float32(
            F_raw, force_factor, path=path, variable_name="forces"
        )

        dimensions = {
            name: int(len(dimension)) for name, dimension in dataset.dimensions.items()
        }
        variable_dimensions = {
            name: list(dataset.variables[name].dimensions)
            for name in required
        }

    active_counts = {
        str(state_label): int(np.count_nonzero(active_state == state_label))
        for state_label in sorted(state_axis_by_label)
    }
    state_mapping = [
        {
            "axis_index": int(axis),
            "source_label": int(source_label),
            "state_label": int(source_label - state_base),
        }
        for axis, source_label in enumerate(state_raw)
    ]
    record_metadata = {
        "mol_id": mol_id,
        "path": path.as_posix(),
        "dimensions": dimensions,
        "variable_dimensions": variable_dimensions,
        "source_frame_count": int(nframes),
        "source_state_count": int(nstates),
        "natoms": int(len(z)),
        "atomic_numbers_source": z_source,
        "original_units": {
            "R": position_unit,
            "E": energy_unit,
            "F": force_unit,
        },
        "output_units": dict(_OUTPUT_UNITS),
        "conversion_factors": {
            "R": float(position_factor),
            "E": float(energy_factor),
            "F": float(force_factor),
        },
        "state_coordinate_base": int(state_base),
        "active_state_base": int(active_base),
        "state_mapping": state_mapping,
        "active_state_counts": active_counts,
    }
    return _NetCDFRecord(
        mol_id=mol_id,
        path=path,
        R=R,
        F=F,
        E=E,
        z=z,
        active_state=active_state,
        state_axis_by_label=state_axis_by_label,
        metadata=record_metadata,
    )


def _read_axis_normalized(
    variable: Any,
    canonical_dimensions: tuple[str, ...],
    *,
    path: Path,
    variable_name: str,
) -> np.ndarray:
    actual_dimensions = tuple(variable.dimensions)
    if (
        len(actual_dimensions) != len(canonical_dimensions)
        or len(set(actual_dimensions)) != len(actual_dimensions)
        or set(actual_dimensions) != set(canonical_dimensions)
    ):
        raise ValueError(
            f"{path}: {variable_name} must use exactly the dimensions "
            f"{canonical_dimensions}, got {actual_dimensions}."
        )

    raw = variable[:]
    masked = np.ma.asarray(raw, dtype=np.float64)
    array = np.asarray(masked.filled(np.nan), dtype=np.float64)
    permutation = tuple(actual_dimensions.index(name) for name in canonical_dimensions)
    array = np.transpose(array, permutation)
    if not np.all(np.isfinite(array)):
        invalid = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"{path}: {variable_name} contains {invalid} masked or non-finite values."
        )
    return array


def _read_atomic_numbers(
    dataset: Any,
    path: Path,
    natoms: int,
) -> tuple[np.ndarray, str]:
    if "atNums" in dataset.variables:
        raw = _read_axis_normalized(
            dataset.variables["atNums"],
            ("atom",),
            path=path,
            variable_name="atNums",
        )
        if raw.shape != (natoms,):
            raise ValueError(
                f"{path}: atNums must have shape {(natoms,)}, got {raw.shape}."
            )
        if not np.all(raw == np.rint(raw)):
            raise ValueError(f"{path}: atNums must contain integer atomic numbers.")
        z = raw.astype(np.int64)
        if np.any((z < 1) | (z > len(_ELEMENT_SYMBOLS))):
            raise ValueError(f"{path}: atNums contains invalid atomic numbers {z.tolist()}.")
        return z, "atNums"

    if "atNames" not in dataset.variables:
        raise ValueError(f"{path}: expected atNums or atNames in the root dataset.")
    names = _read_atom_names(dataset.variables["atNames"], path, natoms)
    numbers: list[int] = []
    for atom_index, name in enumerate(names):
        match = re.match(r"^\s*([A-Za-z]{1,2})", name)
        if match is None:
            raise ValueError(
                f"{path}: cannot parse an element from atNames[{atom_index}]={name!r}."
            )
        letters = match.group(1)
        symbol = letters[0].upper() + letters[1:].lower()
        atomic_number = _ATOMIC_NUMBERS.get(symbol)
        if atomic_number is None:
            raise ValueError(
                f"{path}: unknown element {symbol!r} parsed from "
                f"atNames[{atom_index}]={name!r}."
            )
        numbers.append(atomic_number)
    return np.asarray(numbers, dtype=np.int64), "atNames"


def _read_atom_names(variable: Any, path: Path, natoms: int) -> list[str]:
    dimensions = tuple(variable.dimensions)
    if not dimensions or dimensions[0] != "atom":
        raise ValueError(
            f"{path}: atNames must have atom as its first dimension, got {dimensions}."
        )
    values = variable[:]
    if np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{path}: atNames contains masked entries.")
    array = np.asarray(values)
    if array.shape[0] != natoms:
        raise ValueError(
            f"{path}: atNames must contain {natoms} atoms, got shape {array.shape}."
        )

    if array.ndim == 1:
        entries = list(array)
    elif array.dtype.kind in {"S", "U"}:
        entries = [
            b"".join(np.asarray(row).reshape(-1).tolist())
            if array.dtype.kind == "S"
            else "".join(np.asarray(row).reshape(-1).tolist())
            for row in array
        ]
    else:
        raise ValueError(
            f"{path}: unsupported atNames representation with shape {array.shape} "
            f"and dtype {array.dtype}."
        )

    decoded = []
    for entry in entries:
        if isinstance(entry, (bytes, np.bytes_)):
            decoded.append(bytes(entry).decode("utf-8").rstrip("\x00"))
        else:
            decoded.append(str(entry).rstrip("\x00"))
    return decoded


def _read_state_coordinate(
    dataset: Any,
    path: Path,
    nstates: int,
) -> tuple[np.ndarray, int]:
    if "state" not in dataset.variables:
        return np.arange(nstates, dtype=np.int64), 0

    raw = _read_axis_normalized(
        dataset.variables["state"],
        ("state",),
        path=path,
        variable_name="state",
    )
    if raw.shape != (nstates,) or not np.all(raw == np.rint(raw)):
        raise ValueError(
            f"{path}: state coordinates must be {nstates} integer labels, got {raw}."
        )
    labels = raw.astype(np.int64)
    label_set = set(labels.tolist())
    if label_set == set(range(nstates)):
        base = 0
    elif label_set == set(range(1, nstates + 1)):
        base = 1
    else:
        raise ValueError(
            f"{path}: state coordinates must be a permutation of either "
            f"0..{nstates - 1} or 1..{nstates}, got {labels.tolist()}."
        )
    return labels, base


def _normalize_active_state(
    raw: np.ndarray,
    *,
    nstates: int,
    preferred_base: int,
    path: Path,
) -> tuple[np.ndarray, int]:
    if not np.all(raw == np.rint(raw)):
        raise ValueError(f"{path}: astate must contain integer state labels.")
    labels = raw.astype(np.int64)
    valid_bases = [
        base
        for base in (0, 1)
        if np.all((labels >= base) & (labels < base + nstates))
    ]
    if not valid_bases:
        unique = np.unique(labels).tolist()
        raise ValueError(
            f"{path}: astate labels {unique} are neither zero- nor one-based for "
            f"{nstates} states."
        )
    base = preferred_base if preferred_base in valid_bases else valid_bases[0]
    normalized = labels - base
    return normalized.astype(np.int64), base


def _unquote_unit(raw_unit: Any) -> str:
    if isinstance(raw_unit, bytes):
        value = raw_unit.decode("utf-8")
    else:
        value = str(raw_unit)
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def _unit_key(unit: str) -> str:
    return (
        unit.casefold()
        .replace("\u212b", "å")
        .replace("ångström", "angstrom")
        .replace("ångstrom", "angstrom")
        .replace("angstroem", "angstrom")
        .replace("electron-volts", "electronvolts")
        .replace("electron-volt", "electronvolt")
        .replace("atomic units", "atomicunits")
        .replace("atomic unit", "atomicunit")
        .replace(" ", "")
        .replace("_", "")
        .replace("·", "")
        .replace("*", "")
    )


def _unit_conversion(variable: Any, *, quantity: str, path: Path) -> tuple[str, float]:
    if "units" not in variable.ncattrs():
        raise ValueError(f"{path}: {variable.name} is missing its required units attribute.")
    original = _unquote_unit(variable.getncattr("units"))
    if not original:
        raise ValueError(f"{path}: {variable.name} has an empty units attribute.")
    key = _unit_key(original)

    if quantity == "R":
        converted = {
            "bohr": BOHR_TO_ANGSTROM,
            "bohrs": BOHR_TO_ANGSTROM,
            "a0": BOHR_TO_ANGSTROM,
            "au": BOHR_TO_ANGSTROM,
            "atomicunit": BOHR_TO_ANGSTROM,
            "atomicunits": BOHR_TO_ANGSTROM,
            "å": 1.0,
            "a": 1.0,
            "ang": 1.0,
            "angstrom": 1.0,
            "angstroms": 1.0,
            "ampere": 1.0,
            "amperes": 1.0,
        }
    elif quantity == "E":
        converted = {
            "hartree": HARTREE_TO_EV,
            "hartrees": HARTREE_TO_EV,
            "ha": HARTREE_TO_EV,
            "eh": HARTREE_TO_EV,
            "au": HARTREE_TO_EV,
            "ev": 1.0,
            "electronvolt": 1.0,
            "electronvolts": 1.0,
        }
    elif quantity == "F":
        converted = {
            "hartree/bohr": HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "hartrees/bohr": HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "ha/bohr": HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "eh/bohr": HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "hartreebohr^-1": HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "hartreebohr−1": HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "ev/å": 1.0,
            "ev/a": 1.0,
            "ev/ang": 1.0,
            "ev/angstrom": 1.0,
            "ev/angstroms": 1.0,
            "evå^-1": 1.0,
            "eva^-1": 1.0,
            "evangstrom^-1": 1.0,
            "evangstrom−1": 1.0,
        }
    else:  # pragma: no cover - internal programming error
        raise AssertionError(f"Unknown unit quantity {quantity!r}.")

    factor = converted.get(key)
    if factor is None:
        raise ValueError(
            f"{path}: unsupported units {original!r} for variable {variable.name}; "
            f"expected {_OUTPUT_UNITS[quantity]}."
        )
    return original, factor


def _convert_to_float32(
    array: np.ndarray,
    factor: float,
    *,
    path: Path,
    variable_name: str,
) -> np.ndarray:
    converted = np.asarray(array * factor, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        invalid = int(np.count_nonzero(~np.isfinite(converted)))
        raise ValueError(
            f"{path}: converting {variable_name} to float32 produced "
            f"{invalid} non-finite values."
        )
    return np.ascontiguousarray(converted)


__all__ = [
    "BOHR_TO_ANGSTROM",
    "HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
    "HARTREE_TO_EV",
    "NetCDFSource",
    "NpyDirectorySource",
    "NpzFileSource",
    "StateSource",
]
