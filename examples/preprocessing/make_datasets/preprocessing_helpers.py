from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .graph_utils import all_directed_nonself_pairs, build_sample_pair_tensors


BOND_PROB_CHANNELS = ("single", "aromatic", "double", "triple")
BOND_ORDER_LEVELS = np.asarray([1.0, 1.5, 2.0, 3.0], dtype=np.float32)


class MissingDirectedBondError(ValueError):
    """Raised when a fixed bond falls outside the frame-specific cutoff graph."""


@dataclass(frozen=True)
class StateData:
    """Source-neutral arrays for one molecule and zero-based electronic state."""

    mol_id: str
    state_label: int
    path: Path
    R: np.ndarray
    F: np.ndarray
    E: np.ndarray
    z_per_frame: np.ndarray
    node_mask: np.ndarray
    frame_idx: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )

    def __post_init__(self) -> None:
        """Supply legacy callers with the old implicit row-index provenance."""
        frame_idx = np.asarray(self.frame_idx, dtype=np.int64)
        if frame_idx.size == 0 and self.E.shape[0]:
            frame_idx = np.arange(self.E.shape[0], dtype=np.int64)
        object.__setattr__(self, "frame_idx", frame_idx)

    @property
    def nframes(self) -> int:
        return int(self.E.shape[0])

    @property
    def natoms(self) -> int:
        return int(self.R.shape[1])

    @property
    def z(self) -> np.ndarray:
        return np.asarray(self.z_per_frame[0], dtype=np.int64)


# Retain the original public name for callers outside the refactored builders.
FilteredStateData = StateData


@dataclass(frozen=True)
class BondSpec:
    z: np.ndarray
    bond_idx_i: np.ndarray
    bond_idx_j: np.ndarray
    bond_prob: np.ndarray


def discover_molecule_ids(root: str | Path, requested: list[str] | None = None) -> list[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"No such filtered dataset directory: {root_path}")

    available = sorted(path.name for path in root_path.iterdir() if path.is_dir())
    if requested is None:
        return available

    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(
            f"Requested molecule directories not found below {root_path}: {missing}"
        )
    return list(requested)


def state_dir(root: str | Path, mol_id: str, state_label: int) -> Path:
    return Path(root) / mol_id / f"state_{int(state_label) + 1}.npz_FILES"


def load_filtered_state(
    root: str | Path,
    mol_id: str,
    state_label: int,
    *,
    numframes: int | None = None,
) -> StateData:
    path = state_dir(root, mol_id, state_label)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing state directory for {mol_id} state {state_label}: {path}"
        )

    arrays = {
        "R": np.load(path / "atomic_position.npy").astype(np.float32),
        "F": np.load(path / "force.npy").astype(np.float32),
        "E": np.load(path / "energy.npy").astype(np.float32).reshape(-1),
        "z_per_frame": np.load(path / "atomic_type.npy").astype(np.int64),
    }
    node_mask_path = path / "node_mask.npy"
    if node_mask_path.is_file():
        arrays["node_mask"] = np.load(node_mask_path).astype(bool)
    else:
        arrays["node_mask"] = arrays["z_per_frame"] > 0
    arrays["frame_idx"] = np.arange(arrays["E"].shape[0], dtype=np.int64)

    validate_state_arrays(mol_id=mol_id, state_label=state_label, arrays=arrays)

    n_available = int(arrays["E"].shape[0])
    if numframes is None:
        n_keep = n_available
    else:
        n_keep = int(numframes)
        if n_keep < 0:
            raise ValueError(f"numframes must be non-negative, got {numframes}.")
        if n_keep > n_available:
            raise ValueError(
                f"{mol_id} state {state_label}: requested numframes={n_keep}, "
                f"but only {n_available} frames are available."
            )

    return StateData(
        mol_id=str(mol_id),
        state_label=int(state_label),
        path=path,
        R=arrays["R"][:n_keep],
        F=arrays["F"][:n_keep],
        E=arrays["E"][:n_keep],
        z_per_frame=arrays["z_per_frame"][:n_keep],
        node_mask=arrays["node_mask"][:n_keep],
        frame_idx=arrays["frame_idx"][:n_keep],
    )


def validate_state_arrays(
    *,
    mol_id: str,
    state_label: int,
    arrays: dict[str, np.ndarray],
) -> None:
    R = arrays["R"]
    F = arrays["F"]
    E = arrays["E"]
    z = arrays["z_per_frame"]
    node_mask = arrays["node_mask"]
    frame_idx = arrays.get("frame_idx")

    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"{mol_id} state {state_label}: R must be [T,N,3], got {R.shape}.")
    if F.shape != R.shape:
        raise ValueError(
            f"{mol_id} state {state_label}: force must match R, got {F.shape} and {R.shape}."
        )
    if E.shape != (R.shape[0],):
        raise ValueError(
            f"{mol_id} state {state_label}: energy must be [T], got {E.shape} for T={R.shape[0]}."
        )
    if z.shape != R.shape[:2]:
        raise ValueError(
            f"{mol_id} state {state_label}: atomic_type must be [T,N], got {z.shape}."
        )
    if node_mask.shape != R.shape[:2]:
        raise ValueError(
            f"{mol_id} state {state_label}: node_mask must be [T,N], got {node_mask.shape}."
        )
    if frame_idx is not None and np.asarray(frame_idx).shape != (R.shape[0],):
        raise ValueError(
            f"{mol_id} state {state_label}: frame_idx must be [T], "
            f"got {np.asarray(frame_idx).shape}."
        )
    if z.shape[0] and not np.all(z == z[0]):
        raise ValueError(
            f"{mol_id} state {state_label}: atomic_type changes across frames; "
            "the builders expect a fixed atom ordering."
        )


def pad_atomwise(arr: np.ndarray, max_atoms: int, fill_value: float | int | bool = 0) -> np.ndarray:
    natoms = int(arr.shape[0])
    if natoms > max_atoms:
        raise ValueError(f"Cannot pad {natoms} atoms into max_atoms={max_atoms}.")
    out = np.full((max_atoms,) + arr.shape[1:], fill_value, dtype=arr.dtype)
    out[:natoms] = arr
    return out


def build_pair_slot_lookup(max_atoms: int, base_i: np.ndarray, base_j: np.ndarray) -> np.ndarray:
    lookup = np.full((max_atoms, max_atoms), -1, dtype=np.int64)
    lookup[base_i, base_j] = np.arange(base_i.shape[0], dtype=np.int64)
    return lookup


def bond_order_scalar_to_prob(bond_order: float) -> np.ndarray:
    bond_order = float(bond_order)
    prob = np.zeros((len(BOND_PROB_CHANNELS),), dtype=np.float32)
    if bond_order <= 0.0:
        return prob
    if bond_order <= float(BOND_ORDER_LEVELS[0]):
        prob[0] = 1.0
        return prob
    for low_idx in range(len(BOND_ORDER_LEVELS) - 1):
        low = float(BOND_ORDER_LEVELS[low_idx])
        high = float(BOND_ORDER_LEVELS[low_idx + 1])
        if bond_order <= high:
            high_weight = (bond_order - low) / (high - low)
            prob[low_idx] = 1.0 - high_weight
            prob[low_idx + 1] = high_weight
            return prob
    prob[-1] = 1.0
    return prob


def bond_order_array_to_prob(bond_order: Any) -> np.ndarray:
    entries = list(bond_order) if isinstance(bond_order, (list, tuple)) else list(np.asarray(bond_order, dtype=object))
    descriptors = []
    for entry in entries:
        entry_arr = np.asarray(entry, dtype=np.float32)
        if entry_arr.ndim == 0:
            descriptors.append(bond_order_scalar_to_prob(float(entry_arr)))
            continue
        if entry_arr.shape != (len(BOND_PROB_CHANNELS),):
            raise ValueError(
                "Each bond descriptor entry must be a scalar bond order or a "
                f"{len(BOND_PROB_CHANNELS)}-channel vector, got {entry_arr.shape}."
            )
        if np.any(entry_arr < 0.0):
            raise ValueError("Bond probability vectors must be non-negative.")
        total = float(entry_arr.sum())
        if total > 0.0 and not np.isclose(total, 1.0, atol=2e-2):
            raise ValueError(
                f"Bond probability vectors must sum to 1.0 when nonzero, got {total:.6f}."
            )
        descriptors.append(entry_arr.astype(np.float32))
    return np.stack(descriptors, axis=0).astype(np.float32)


def load_bond_specs(path: str | Path) -> dict[tuple[str, int], BondSpec]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if payload.get("bond_prob_channels") != list(BOND_PROB_CHANNELS):
        raise ValueError(
            f"{path}: bond_prob_channels must be {list(BOND_PROB_CHANNELS)!r}."
        )

    specs: dict[tuple[str, int], BondSpec] = {}
    for entry in payload["specs"]:
        key = (str(entry["mol_id"]), int(entry["state"]))
        if key in specs:
            raise ValueError(f"{path}: duplicate bond spec for molecule/state {key}.")
        specs[key] = BondSpec(
            z=np.asarray(entry["z"], dtype=np.int64),
            bond_idx_i=np.asarray(entry["bond_idx_i"], dtype=np.int64),
            bond_idx_j=np.asarray(entry["bond_idx_j"], dtype=np.int64),
            bond_prob=bond_order_array_to_prob(entry["bond_prob"]),
        )
    return specs


def validate_bond_spec_against_state(data: FilteredStateData, spec: BondSpec) -> None:
    if spec.z.shape[0] != data.natoms:
        raise ValueError(
            f"{data.mol_id} state {data.state_label}: bond spec has {spec.z.shape[0]} atoms, "
            f"but filtered arrays have {data.natoms}."
        )
    if not np.array_equal(spec.z, data.z):
        raise ValueError(
            f"{data.mol_id} state {data.state_label}: atom order/z mismatch.\n"
            f"z from filtered arrays: {data.z.tolist()}\n"
            f"z from bond spec:       {spec.z.tolist()}"
        )
    if spec.bond_idx_i.shape != spec.bond_idx_j.shape:
        raise ValueError(f"{data.mol_id} state {data.state_label}: bond index arrays differ.")
    if spec.bond_prob.shape[:-1] != spec.bond_idx_i.shape or spec.bond_prob.shape[-1] != 4:
        raise ValueError(
            f"{data.mol_id} state {data.state_label}: bond_prob must align with bond indices "
            "and have four channels."
        )
    if spec.bond_idx_i.size > 0:
        if spec.bond_idx_i.min() < 0 or spec.bond_idx_j.min() < 0:
            raise ValueError(f"{data.mol_id} state {data.state_label}: negative bond index found.")
        if spec.bond_idx_i.max() >= data.natoms or spec.bond_idx_j.max() >= data.natoms:
            raise ValueError(f"{data.mol_id} state {data.state_label}: bond index exceeds natoms.")


def build_bond_annotations(
    *,
    pair_mask: np.ndarray,
    R: np.ndarray,
    spec: BondSpec,
    pair_slot_lookup: np.ndarray,
    mol_id: str,
    state_label: int,
    frame_idx: int,
    r_cut: float,
) -> tuple[np.ndarray, np.ndarray]:
    bond_prob = np.zeros((pair_mask.shape[0], len(BOND_PROB_CHANNELS)), dtype=np.float32)
    bond_mask = np.zeros((pair_mask.shape[0],), dtype=bool)

    for edge_pos, (i, j, prob) in enumerate(zip(spec.bond_idx_i, spec.bond_idx_j, spec.bond_prob)):
        slot = int(pair_slot_lookup[int(i), int(j)])
        if slot < 0:
            raise ValueError(
                f"{mol_id} state {state_label}: directed bond ({int(i)} -> {int(j)}) "
                "does not map to any cutoff-graph slot."
            )
        if not bool(pair_mask[slot]):
            distance = float(np.linalg.norm(R[int(i)] - R[int(j)]))
            raise MissingDirectedBondError(
                f"mol={mol_id} state={state_label} frame_idx={frame_idx} "
                f"edge=({int(i)} -> {int(j)}) distance={distance:.6f} angstrom "
                f"is missing from the cutoff graph for r_cut={float(r_cut):.3f} angstrom."
            )
        if bond_mask[slot]:
            raise ValueError(
                f"{mol_id} state {state_label}: duplicate directed bond edge "
                f"({int(i)} -> {int(j)}) at bond list position {edge_pos}."
            )
        bond_prob[slot] = np.asarray(prob, dtype=np.float32)
        bond_mask[slot] = True

    return bond_prob, bond_mask


def load_state_grid(
    root: str | Path,
    molecule_ids: list[str],
    states: list[int],
    *,
    numframes: int | None = None,
) -> list[FilteredStateData]:
    return [
        load_filtered_state(root, mol_id, state, numframes=numframes)
        for mol_id in molecule_ids
        for state in states
    ]


def balanced_frames_per_molecule(
    available_frames: dict[str, int],
    numframes: int | None,
) -> int | None:
    """Return the common per-molecule frame count for a total frame request.

    ``numframes`` is interpreted as the requested size of the complete dataset,
    not as a per-molecule limit. Any remainder from dividing by the number of
    molecules is dropped. If one molecule cannot provide its share, every
    molecule is reduced to that molecule's available count so the dataset stays
    balanced.
    """
    if numframes is None:
        return None
    if not available_frames:
        raise ValueError("Cannot allocate numframes without any molecules.")

    requested = int(numframes)
    if requested <= 0:
        raise ValueError(f"numframes must be positive, got {numframes}.")

    per_molecule = requested // len(available_frames)
    if per_molecule == 0:
        raise ValueError(
            f"numframes={requested} is smaller than the number of molecules "
            f"({len(available_frames)}); at least one frame per molecule is required."
        )

    normalized_available = {
        str(mol_id): int(count) for mol_id, count in available_frames.items()
    }
    if any(count < 0 for count in normalized_available.values()):
        raise ValueError(
            f"Available frame counts must be non-negative: {normalized_available}."
        )
    smallest_available = min(normalized_available.values())
    if smallest_available == 0:
        empty_molecules = [
            mol_id for mol_id, count in normalized_available.items() if count == 0
        ]
        raise ValueError(
            "Cannot build a balanced dataset because these molecules have no "
            f"available frames: {empty_molecules}."
        )
    return min(per_molecule, smallest_available)


def take_state_frames(data: StateData, nframes: int) -> StateData:
    """Return a view-like state record containing its first ``nframes`` rows."""
    nframes = int(nframes)
    if nframes < 0 or nframes > data.nframes:
        raise ValueError(
            f"{data.mol_id} state {data.state_label}: cannot take {nframes} "
            f"of {data.nframes} available frames."
        )
    return replace(
        data,
        R=data.R[:nframes],
        F=data.F[:nframes],
        E=data.E[:nframes],
        z_per_frame=data.z_per_frame[:nframes],
        node_mask=data.node_mask[:nframes],
        frame_idx=data.frame_idx[:nframes],
    )


def _allocate_evenly_with_capacity(total: int, capacities: list[int]) -> list[int]:
    """Split ``total`` nearly evenly while respecting per-entry capacities."""
    allocations = [0] * len(capacities)
    remaining = int(total)
    while remaining:
        active = [
            index
            for index, capacity in enumerate(capacities)
            if allocations[index] < capacity
        ]
        if not active:
            raise ValueError(
                f"Cannot allocate {total} frames within capacities {capacities}."
            )
        share = max(1, remaining // len(active))
        for index in active:
            grant = min(share, capacities[index] - allocations[index], remaining)
            allocations[index] += grant
            remaining -= grant
            if remaining == 0:
                break
    return allocations


def balance_state_data_by_molecule(
    state_data: list[StateData],
    molecule_ids: list[str],
    numframes: int | None,
) -> tuple[list[StateData], int | None]:
    """Apply a total frame limit equally across molecules and their states.

    State-expanded builders count every state row as one dataset frame. A
    molecule's allocation is therefore divided as evenly as possible among its
    selected states, subject to the frames available in each state.
    """
    if numframes is None:
        return list(state_data), None

    ordered_molecules = list(dict.fromkeys(str(mol_id) for mol_id in molecule_ids))
    if len(ordered_molecules) != len(molecule_ids):
        raise ValueError(f"Molecule IDs must be unique, got {molecule_ids}.")

    entries_by_molecule = {
        mol_id: [data for data in state_data if data.mol_id == mol_id]
        for mol_id in ordered_molecules
    }
    available_frames = {
        mol_id: sum(data.nframes for data in entries)
        for mol_id, entries in entries_by_molecule.items()
    }
    per_molecule = balanced_frames_per_molecule(available_frames, numframes)
    assert per_molecule is not None

    allocations_by_entry_id: dict[int, int] = {}
    for entries in entries_by_molecule.values():
        allocations = _allocate_evenly_with_capacity(
            per_molecule,
            [data.nframes for data in entries],
        )
        allocations_by_entry_id.update(
            (id(data), allocation)
            for data, allocation in zip(entries, allocations, strict=True)
        )

    balanced = []
    for data in state_data:
        n_keep = allocations_by_entry_id.get(id(data), 0)
        if n_keep:
            balanced.append(take_state_frames(data, n_keep))
    return balanced, per_molecule


def select_bonded_state_data(
    state_data: list[StateData],
    bond_specs: dict[tuple[str, int], BondSpec],
    *,
    skip_missing_bond_specs: bool,
) -> list[FilteredStateData]:
    selected = []
    missing = []
    for data in state_data:
        key = (data.mol_id, data.state_label)
        spec = bond_specs.get(key)
        if spec is None:
            missing.append(key)
            continue
        validate_bond_spec_against_state(data, spec)
        selected.append(data)

    if missing and not skip_missing_bond_specs:
        raise KeyError(
            "Missing bond specs for molecule/state entries: "
            f"{missing}. Add them to the YAML or pass --skip-missing-bond-specs."
        )
    if missing:
        warnings.warn(
            f"Skipping {len(missing)} molecule/state entries with missing bond specs: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )
    return selected


def write_npz_and_metadata(output: str | Path, save_dict: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    requested_path = Path(output).expanduser()
    if requested_path.suffix != ".npz":
        requested_path = Path(f"{requested_path}.npz")
    out_path = requested_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path.as_posix(), **save_dict)
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved dataset to:  {out_path}")
    print(f"Saved metadata to: {meta_path}")
    print("\nSaved arrays:")
    for key, value in save_dict.items():
        print(f"{key:28s} shape={value.shape} dtype={value.dtype}")


def base_graph_context(max_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    base_i, base_j = all_directed_nonself_pairs(max_atoms)
    pair_slot_lookup = build_pair_slot_lookup(max_atoms, base_i, base_j)
    return base_i, base_j, pair_slot_lookup, int(base_i.shape[0])


def state_data_summary(state_data: list[StateData]) -> list[dict[str, Any]]:
    return [
        {
            "mol_id": data.mol_id,
            "state": int(data.state_label),
            "path": data.path.as_posix(),
            "nframes": int(data.nframes),
            "natoms": int(data.natoms),
            "z": data.z.tolist(),
            "source_frame_start": (
                None if data.nframes == 0 else int(data.frame_idx[0])
            ),
            "source_frame_end": (
                None if data.nframes == 0 else int(data.frame_idx[-1])
            ),
        }
        for data in state_data
    ]
