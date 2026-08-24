"""Dataset assembly for the MLFF preprocessing examples."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data_sources import StateSource
from .preprocessing_helpers import (
    BOND_PROB_CHANNELS,
    MissingDirectedBondError,
    StateData,
    balance_state_data_by_molecule,
    balanced_frames_per_molecule,
    base_graph_context,
    build_bond_annotations,
    build_sample_pair_tensors,
    load_bond_specs,
    pad_atomwise,
    select_bonded_state_data,
    state_data_summary,
    take_state_frames,
    validate_bond_spec_against_state,
    write_npz_and_metadata,
)


GROUND_STATE = 0
DELTA_STATE_1 = 1
DELTA_STATE_2 = 2

SUPPORTED_EXCITED_STATES = (1, 2)


def _load_state_grid(
    source: StateSource,
    molecule_ids: Iterable[str],
    states: Iterable[int],
    *,
    active_only: bool,
) -> list[StateData]:
    return [
        source.load_state(mol_id, int(state), active_only=active_only)
        for mol_id in molecule_ids
        for state in states
    ]


def _apply_metadata_overrides(
    meta: dict[str, Any],
    metadata_overrides: dict[str, Any] | None,
    *,
    protected_keys: Iterable[str],
) -> dict[str, Any]:
    """Apply source metadata without allowing it to stale runtime-derived facts."""
    if not metadata_overrides:
        return meta
    protected = {key: meta[key] for key in protected_keys if key in meta}
    meta.update(metadata_overrides)
    meta.update(protected)
    return meta


def assert_delta_geometry_alignment(
    *,
    mol_id: str,
    R0: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    atol: float,
) -> None:
    max_01 = float(np.max(np.abs(R1 - R0))) if R0.size else 0.0
    max_02 = float(np.max(np.abs(R2 - R0))) if R0.size else 0.0
    if max(max_01, max_02) <= float(atol):
        return
    raise ValueError(
        f"{mol_id}: delta row-index pairing failed geometry check. "
        f"max|R_state2-R_state1|={max_01:.6g}, "
        f"max|R_state3-R_state1|={max_02:.6g}, atol={atol}. "
        "These filtered state directories do not appear row-aligned."
    )


def validate_excited_states(states: list[int]) -> tuple[int, ...]:
    normalized = tuple(int(state) for state in states)
    if not normalized:
        raise ValueError("Delta-offset construction requires at least one excited state.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate excited states are not allowed: {normalized}.")
    invalid = sorted(set(normalized) - set(SUPPORTED_EXCITED_STATES))
    if invalid:
        raise ValueError(
            "Delta-offset datasets contain active S1/S2 rows only; "
            f"unsupported zero-based states: {invalid}."
        )
    return tuple(sorted(normalized))


def validate_active_labels(state_data: Iterable[StateData]) -> None:
    for data in state_data:
        if not np.isfinite(data.E).all():
            raise ValueError(
                f"{data.mol_id} state {data.state_label}: "
                "active energies contain non-finite values."
            )
        real_force = data.F[np.asarray(data.node_mask, dtype=bool)]
        if not np.isfinite(real_force).all():
            raise ValueError(
                f"{data.mol_id} state {data.state_label}: active forces contain "
                "non-finite values on non-padded atoms."
            )
        real_position = data.R[np.asarray(data.node_mask, dtype=bool)]
        if not np.isfinite(real_position).all():
            raise ValueError(
                f"{data.mol_id} state {data.state_label}: positions contain non-finite "
                "values on non-padded atoms."
            )


def trim_trailing_padding(state_data: Iterable[StateData]) -> list[StateData]:
    """Remove source-level trailing padding before constructing pair graphs."""
    trimmed = []
    for data in state_data:
        if data.nframes == 0:
            trimmed.append(data)
            continue
        masks = np.asarray(data.node_mask, dtype=bool)
        first_mask = masks[0]
        if not np.array_equal(masks, np.broadcast_to(first_mask, masks.shape)):
            raise ValueError(
                f"{data.mol_id} state {data.state_label}: node_mask changes across frames."
            )
        natoms = int(first_mask.sum())
        expected_mask = np.arange(data.natoms) < natoms
        if not np.array_equal(first_mask, expected_mask):
            raise ValueError(
                f"{data.mol_id} state {data.state_label}: real atoms must precede "
                "trailing padded atoms in node_mask."
            )
        if natoms == 0:
            raise ValueError(
                f"{data.mol_id} state {data.state_label}: no real atoms were selected."
            )
        if natoms == data.natoms:
            trimmed.append(data)
            continue
        trimmed.append(
            replace(
                data,
                R=data.R[:, :natoms],
                F=data.F[:, :natoms],
                z_per_frame=data.z_per_frame[:, :natoms],
                node_mask=data.node_mask[:, :natoms],
            )
        )
    return trimmed


def normalize_bond_annotations(
    bond_prob: np.ndarray,
    bond_mask: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize annotated probabilities to the strict MLFF graph contract."""
    normalized = np.asarray(bond_prob, dtype=np.float32).copy()
    mask = np.asarray(bond_mask, dtype=bool)
    if not np.isfinite(normalized).all() or (normalized < 0).any():
        raise ValueError(f"{context}: bond probabilities must be finite and non-negative.")
    totals = normalized.sum(axis=-1)
    if (totals[mask] <= 0).any():
        raise ValueError(
            f"{context}: every annotated bond must have positive probability mass."
        )
    normalized[mask] /= totals[mask, None]
    return normalized, mask


def select_state_data_for_bonds(
    state_data: Iterable[StateData],
    bond_specs: dict,
    *,
    skip_missing_bond_specs: bool,
) -> tuple[list[StateData], list[dict[str, Any]]]:
    selected = []
    skipped = []
    for data in state_data:
        required_states = [GROUND_STATE, data.state_label]
        missing_states = [
            state
            for state in required_states
            if (data.mol_id, state) not in bond_specs
        ]
        if missing_states:
            record = {
                "mol_id": data.mol_id,
                "active_state": int(data.state_label),
                "missing_bond_spec_states": [int(state) for state in missing_states],
            }
            if not skip_missing_bond_specs:
                raise KeyError(
                    f"{data.mol_id} active state {data.state_label}: missing bond specs "
                    f"for zero-based states {missing_states}. Add them to the YAML or pass "
                    "--skip-missing-bond-specs."
                )
            skipped.append(record)
            warnings.warn(
                f"Skipping molecule/state entry with missing bond specs: {record}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        validate_bond_spec_against_state(
            data,
            bond_specs[(data.mol_id, data.state_label)],
        )
        validate_bond_spec_against_state(
            data,
            bond_specs[(data.mol_id, GROUND_STATE)],
        )
        selected.append(data)
    return selected, skipped


def assemble_so3krates_no_bond_dataset(
    *,
    source: StateSource,
    output: str | Path,
    molecules: list[str] | None,
    states: list[int],
    numframes: int | None,
    r_cut: float,
    metadata_overrides: dict[str, Any] | None = None,
) -> None:
    molecule_ids = source.discover_molecule_ids(molecules)
    state_data = _load_state_grid(
        source,
        molecule_ids,
        states,
        active_only=False,
    )
    state_data, frames_per_molecule = balance_state_data_by_molecule(
        state_data,
        molecule_ids,
        numframes,
    )

    total_frames = sum(data.nframes for data in state_data)
    max_atoms = max(data.natoms for data in state_data)
    base_i, base_j, _, max_pairs = base_graph_context(max_atoms)

    r_all = np.zeros((total_frames, max_atoms, 3), dtype=np.float32)
    f_all = np.zeros((total_frames, max_atoms, 3), dtype=np.float32)
    e_all = np.zeros((total_frames,), dtype=np.float32)
    z_all = np.zeros((total_frames, max_atoms), dtype=np.int64)
    atom_state_all = np.zeros((total_frames, max_atoms), dtype=np.int64)
    node_mask_all = np.zeros((total_frames, max_atoms), dtype=bool)
    idx_i_all = np.full((total_frames, max_pairs), -1, dtype=np.int64)
    idx_j_all = np.full((total_frames, max_pairs), -1, dtype=np.int64)
    pair_mask_all = np.zeros((total_frames, max_pairs), dtype=bool)
    n_atoms_all = np.zeros((total_frames,), dtype=np.int64)
    molecule_index_all = np.zeros((total_frames,), dtype=np.int64)
    frame_idx_all = np.zeros((total_frames,), dtype=np.int64)
    astate_all = np.zeros((total_frames,), dtype=np.int64)
    target_state_all = np.zeros((total_frames,), dtype=np.int64)

    molecule_to_index = {mol_id: idx for idx, mol_id in enumerate(molecule_ids)}
    sample_idx = 0
    for data in state_data:
        for local_t in range(data.nframes):
            natoms = data.natoms
            state_label = data.state_label
            r_padded = pad_atomwise(data.R[local_t], max_atoms, 0.0)
            idx_i, idx_j, pair_mask = build_sample_pair_tensors(
                R=r_padded,
                natoms=natoms,
                base_i=base_i,
                base_j=base_j,
                r_cut=r_cut,
            )

            r_all[sample_idx] = r_padded
            f_all[sample_idx] = pad_atomwise(data.F[local_t], max_atoms, 0.0)
            e_all[sample_idx] = data.E[local_t]
            z_all[sample_idx] = pad_atomwise(data.z_per_frame[local_t], max_atoms, 0)
            atom_state_all[sample_idx, :natoms] = state_label
            node_mask_all[sample_idx] = pad_atomwise(
                data.node_mask[local_t], max_atoms, False
            )
            idx_i_all[sample_idx] = idx_i
            idx_j_all[sample_idx] = idx_j
            pair_mask_all[sample_idx] = pair_mask
            n_atoms_all[sample_idx] = natoms
            molecule_index_all[sample_idx] = molecule_to_index[data.mol_id]
            frame_idx_all[sample_idx] = int(data.frame_idx[local_t])
            astate_all[sample_idx] = state_label + 1
            target_state_all[sample_idx] = state_label
            sample_idx += 1

    save_dict = {
        "R": r_all,
        "F": f_all,
        "E": e_all,
        "z": z_all,
        "atom_state": atom_state_all,
        "node_mask": node_mask_all,
        "idx_i": idx_i_all,
        "idx_j": idx_j_all,
        "pair_mask": pair_mask_all,
        "n_atoms": n_atoms_all,
        "molecule_index": molecule_index_all,
        "frame_idx": frame_idx_all,
        "astate": astate_all,
        "target_state": target_state_all,
        "molecule_names": np.asarray(molecule_ids),
    }

    meta = {
        "source": "filtered_shnitsel_data_saad npy directories",
        "builder": "make_so3krates_dataset.py",
        "model_intent": "so3krates without bond_prob or bond_mask descriptors",
        "units": {"R": "angstrom", "E": "eV", "F": "eV/angstrom"},
        "selection_logic": "state-expanded rows from filtered state directories",
        "states": [int(state) for state in states],
        "molecules": molecule_ids,
        "numframes": None if numframes is None else int(numframes),
        "frames_per_molecule": (
            None if frames_per_molecule is None else int(frames_per_molecule)
        ),
        "total_frames": int(total_frames),
        "max_atoms": int(max_atoms),
        "max_pairs": int(max_pairs),
        "r_cut": float(r_cut),
        "state_data": state_data_summary(state_data),
    }
    _apply_metadata_overrides(
        meta,
        metadata_overrides,
        protected_keys=(
            "states",
            "molecules",
            "numframes",
            "frames_per_molecule",
            "total_frames",
            "max_atoms",
            "max_pairs",
            "r_cut",
            "state_data",
        ),
    )
    write_npz_and_metadata(output, save_dict, meta)


def assemble_so3kratex_dataset(
    *,
    source: StateSource,
    output: str | Path,
    molecules: list[str] | None,
    states: list[int],
    numframes: int | None,
    r_cut: float,
    bond_specs_file: str | Path,
    skip_missing_bond_specs: bool,
    metadata_overrides: dict[str, Any] | None = None,
) -> None:
    molecule_ids = source.discover_molecule_ids(molecules)
    loaded_state_data = _load_state_grid(
        source,
        molecule_ids,
        states,
        active_only=False,
    )
    bond_specs = load_bond_specs(bond_specs_file)
    state_data = select_bonded_state_data(
        loaded_state_data,
        bond_specs,
        skip_missing_bond_specs=skip_missing_bond_specs,
    )
    if not state_data:
        raise ValueError("No molecule/state data left after applying bond-spec filtering.")

    kept_molecules = sorted({data.mol_id for data in state_data})
    state_data, frames_per_molecule = balance_state_data_by_molecule(
        state_data,
        kept_molecules,
        numframes,
    )
    molecule_to_index = {mol_id: idx for idx, mol_id in enumerate(kept_molecules)}
    total_requested_frames = sum(data.nframes for data in state_data)
    max_atoms = max(data.natoms for data in state_data)
    base_i, base_j, pair_slot_lookup, max_pairs = base_graph_context(max_atoms)
    max_bond_edges = max(
        int(bond_specs[(data.mol_id, data.state_label)].bond_idx_i.shape[0])
        for data in state_data
    )

    r_all = np.zeros((total_requested_frames, max_atoms, 3), dtype=np.float32)
    f_all = np.zeros((total_requested_frames, max_atoms, 3), dtype=np.float32)
    e_all = np.zeros((total_requested_frames,), dtype=np.float32)
    z_all = np.zeros((total_requested_frames, max_atoms), dtype=np.int64)
    atom_state_all = np.zeros((total_requested_frames, max_atoms), dtype=np.int64)
    node_mask_all = np.zeros((total_requested_frames, max_atoms), dtype=bool)
    idx_i_all = np.full((total_requested_frames, max_pairs), -1, dtype=np.int64)
    idx_j_all = np.full((total_requested_frames, max_pairs), -1, dtype=np.int64)
    pair_mask_all = np.zeros((total_requested_frames, max_pairs), dtype=bool)
    bond_prob_all = np.zeros(
        (total_requested_frames, max_pairs, len(BOND_PROB_CHANNELS)),
        dtype=np.float32,
    )
    bond_mask_all = np.zeros((total_requested_frames, max_pairs), dtype=bool)
    bond_idx_i_all = np.full(
        (total_requested_frames, max_bond_edges), -1, dtype=np.int64
    )
    bond_idx_j_all = np.full(
        (total_requested_frames, max_bond_edges), -1, dtype=np.int64
    )
    bond_reference_local_t_all = np.zeros((total_requested_frames,), dtype=np.int64)
    bond_reference_frame_idx_all = np.zeros((total_requested_frames,), dtype=np.int64)
    n_atoms_all = np.zeros((total_requested_frames,), dtype=np.int64)
    molecule_index_all = np.zeros((total_requested_frames,), dtype=np.int64)
    frame_idx_all = np.zeros((total_requested_frames,), dtype=np.int64)
    astate_all = np.zeros((total_requested_frames,), dtype=np.int64)
    target_state_all = np.zeros((total_requested_frames,), dtype=np.int64)

    sample_idx = 0
    skipped_frames = []
    for data in state_data:
        spec = bond_specs[(data.mol_id, data.state_label)]
        n_bond_edges = int(spec.bond_idx_i.shape[0])
        for local_t in range(data.nframes):
            natoms = data.natoms
            state_label = data.state_label
            source_frame_idx = int(data.frame_idx[local_t])
            r_padded = pad_atomwise(data.R[local_t], max_atoms, 0.0)
            idx_i, idx_j, pair_mask = build_sample_pair_tensors(
                R=r_padded,
                natoms=natoms,
                base_i=base_i,
                base_j=base_j,
                r_cut=r_cut,
            )
            try:
                bond_prob, bond_mask = build_bond_annotations(
                    pair_mask=pair_mask,
                    R=r_padded,
                    spec=spec,
                    pair_slot_lookup=pair_slot_lookup,
                    mol_id=data.mol_id,
                    state_label=state_label,
                    frame_idx=source_frame_idx,
                    r_cut=r_cut,
                )
            except MissingDirectedBondError as exc:
                skipped_frames.append(
                    {
                        "mol_id": data.mol_id,
                        "target_state": int(state_label),
                        "frame_idx": source_frame_idx,
                        "reason": str(exc),
                    }
                )
                warnings.warn(f"Skipping frame due to missing directed bond: {exc}")
                continue

            r_all[sample_idx] = r_padded
            f_all[sample_idx] = pad_atomwise(data.F[local_t], max_atoms, 0.0)
            e_all[sample_idx] = data.E[local_t]
            z_all[sample_idx] = pad_atomwise(data.z_per_frame[local_t], max_atoms, 0)
            atom_state_all[sample_idx, :natoms] = state_label
            node_mask_all[sample_idx] = pad_atomwise(
                data.node_mask[local_t], max_atoms, False
            )
            idx_i_all[sample_idx] = idx_i
            idx_j_all[sample_idx] = idx_j
            pair_mask_all[sample_idx] = pair_mask
            bond_prob_all[sample_idx] = bond_prob
            bond_mask_all[sample_idx] = bond_mask
            bond_idx_i_all[sample_idx, :n_bond_edges] = spec.bond_idx_i
            bond_idx_j_all[sample_idx, :n_bond_edges] = spec.bond_idx_j
            n_atoms_all[sample_idx] = natoms
            molecule_index_all[sample_idx] = molecule_to_index[data.mol_id]
            frame_idx_all[sample_idx] = source_frame_idx
            astate_all[sample_idx] = state_label + 1
            target_state_all[sample_idx] = state_label
            sample_idx += 1

    save_dict = {
        "R": r_all[:sample_idx],
        "F": f_all[:sample_idx],
        "E": e_all[:sample_idx],
        "z": z_all[:sample_idx],
        "atom_state": atom_state_all[:sample_idx],
        "node_mask": node_mask_all[:sample_idx],
        "idx_i": idx_i_all[:sample_idx],
        "idx_j": idx_j_all[:sample_idx],
        "pair_mask": pair_mask_all[:sample_idx],
        "bond_idx_i": bond_idx_i_all[:sample_idx],
        "bond_idx_j": bond_idx_j_all[:sample_idx],
        "bond_prob": bond_prob_all[:sample_idx],
        "bond_mask": bond_mask_all[:sample_idx],
        "bond_reference_local_t": bond_reference_local_t_all[:sample_idx],
        "bond_reference_frame_idx": bond_reference_frame_idx_all[:sample_idx],
        "n_atoms": n_atoms_all[:sample_idx],
        "molecule_index": molecule_index_all[:sample_idx],
        "frame_idx": frame_idx_all[:sample_idx],
        "astate": astate_all[:sample_idx],
        "target_state": target_state_all[:sample_idx],
        "molecule_names": np.asarray(kept_molecules),
    }

    meta = {
        "source": "filtered_shnitsel_data_saad npy directories",
        "builder": "make_so3krateX_dataset.py",
        "model_intent": "so3krateX multi-state model with bond_prob and bond_mask",
        "units": {"R": "angstrom", "E": "eV", "F": "eV/angstrom"},
        "selection_logic": "state-expanded rows from filtered state directories",
        "states": [int(state) for state in states],
        "molecules": kept_molecules,
        "numframes": None if numframes is None else int(numframes),
        "frames_per_molecule": (
            None if frames_per_molecule is None else int(frames_per_molecule)
        ),
        "requested_total_frames": int(total_requested_frames),
        "total_frames": int(sample_idx),
        "skipped_frame_count": int(len(skipped_frames)),
        "skipped_frames": skipped_frames,
        "max_atoms": int(max_atoms),
        "max_pairs": int(max_pairs),
        "max_bond_edges": int(max_bond_edges),
        "r_cut": float(r_cut),
        "bond_prob_channels": list(BOND_PROB_CHANNELS),
        "state_data": state_data_summary(state_data),
    }
    _apply_metadata_overrides(
        meta,
        metadata_overrides,
        protected_keys=(
            "states",
            "molecules",
            "numframes",
            "frames_per_molecule",
            "requested_total_frames",
            "total_frames",
            "skipped_frame_count",
            "skipped_frames",
            "max_atoms",
            "max_pairs",
            "max_bond_edges",
            "r_cut",
            "state_data",
        ),
    )
    write_npz_and_metadata(output, save_dict, meta)


def assemble_delta_dataset(
    *,
    source: StateSource,
    output: str | Path,
    molecules: list[str] | None,
    numframes: int | None,
    r_cut: float,
    bond_specs_file: str | Path,
    skip_missing_bond_specs: bool,
    check_geometry_alignment: bool,
    geometry_atol: float,
    metadata_overrides: dict[str, Any] | None = None,
) -> None:
    molecule_ids = source.discover_molecule_ids(molecules)
    bond_specs = load_bond_specs(bond_specs_file)

    raw_molecule_blocks = []
    skipped_molecules = []
    for mol_id in molecule_ids:
        needed = [
            (mol_id, GROUND_STATE),
            (mol_id, DELTA_STATE_1),
            (mol_id, DELTA_STATE_2),
        ]
        missing = [key for key in needed if key not in bond_specs]
        if missing:
            if skip_missing_bond_specs:
                skipped_molecules.append(
                    {"mol_id": mol_id, "missing_bond_specs": missing}
                )
                warnings.warn(f"Skipping {mol_id}; missing bond specs: {missing}")
                continue
            raise KeyError(
                f"{mol_id}: missing delta bond specs {missing}. "
                "Add them to the YAML or pass --skip-missing-bond-specs."
            )

        raw_states = [
            source.load_state(mol_id, state, active_only=False)
            for state in (GROUND_STATE, DELTA_STATE_1, DELTA_STATE_2)
        ]
        raw_molecule_blocks.append((mol_id, raw_states))

    if not raw_molecule_blocks:
        raise ValueError("No molecules left for delta dataset construction.")

    available_frames = {
        mol_id: min(state.nframes for state in state_rows)
        for mol_id, state_rows in raw_molecule_blocks
    }
    frames_per_molecule = balanced_frames_per_molecule(
        available_frames,
        numframes,
    )

    molecule_blocks = []
    for mol_id, raw_states in raw_molecule_blocks:
        n_keep = (
            available_frames[mol_id]
            if frames_per_molecule is None
            else frames_per_molecule
        )
        state_rows = [take_state_frames(data, n_keep) for data in raw_states]
        for data in state_rows:
            validate_bond_spec_against_state(
                data, bond_specs[(mol_id, data.state_label)]
            )
        if check_geometry_alignment:
            assert_delta_geometry_alignment(
                mol_id=mol_id,
                R0=state_rows[0].R,
                R1=state_rows[1].R,
                R2=state_rows[2].R,
                atol=geometry_atol,
            )
        molecule_blocks.append((mol_id, state_rows, n_keep))

    kept_molecules = [mol_id for mol_id, _, _ in molecule_blocks]
    molecule_to_index = {mol_id: idx for idx, mol_id in enumerate(kept_molecules)}
    total_requested_frames = sum(n_keep for _, _, n_keep in molecule_blocks)
    max_atoms = max(
        state_rows[GROUND_STATE].natoms for _, state_rows, _ in molecule_blocks
    )
    base_i, base_j, pair_slot_lookup, max_pairs = base_graph_context(max_atoms)
    max_bond_edges = max(
        int(bond_specs[(mol_id, state)].bond_idx_i.shape[0])
        for mol_id, _, _ in molecule_blocks
        for state in (GROUND_STATE, DELTA_STATE_1, DELTA_STATE_2)
    )

    r_all = np.zeros((total_requested_frames, max_atoms, 3), dtype=np.float32)
    f_all = np.zeros((total_requested_frames, max_atoms, 3), dtype=np.float32)
    e_all = np.zeros((total_requested_frames,), dtype=np.float32)
    delta_e1_all = np.zeros((total_requested_frames,), dtype=np.float32)
    delta_e2_all = np.zeros((total_requested_frames,), dtype=np.float32)
    delta_f1_all = np.zeros(
        (total_requested_frames, max_atoms, 3), dtype=np.float32
    )
    delta_f2_all = np.zeros(
        (total_requested_frames, max_atoms, 3), dtype=np.float32
    )
    z_all = np.zeros((total_requested_frames, max_atoms), dtype=np.int64)
    atom_state_all = np.zeros((total_requested_frames, max_atoms), dtype=np.int64)
    node_mask_all = np.zeros((total_requested_frames, max_atoms), dtype=bool)
    idx_i_all = np.full((total_requested_frames, max_pairs), -1, dtype=np.int64)
    idx_j_all = np.full((total_requested_frames, max_pairs), -1, dtype=np.int64)
    pair_mask_all = np.zeros((total_requested_frames, max_pairs), dtype=bool)
    bond_prob_by_state = {
        state: np.zeros(
            (total_requested_frames, max_pairs, len(BOND_PROB_CHANNELS)),
            dtype=np.float32,
        )
        for state in (GROUND_STATE, DELTA_STATE_1, DELTA_STATE_2)
    }
    bond_mask_by_state = {
        state: np.zeros((total_requested_frames, max_pairs), dtype=bool)
        for state in (GROUND_STATE, DELTA_STATE_1, DELTA_STATE_2)
    }
    bond_idx_i_all = np.full(
        (total_requested_frames, max_bond_edges), -1, dtype=np.int64
    )
    bond_idx_j_all = np.full(
        (total_requested_frames, max_bond_edges), -1, dtype=np.int64
    )
    bond_reference_local_t_all = np.zeros((total_requested_frames,), dtype=np.int64)
    bond_reference_frame_idx_all = np.zeros((total_requested_frames,), dtype=np.int64)
    n_atoms_all = np.zeros((total_requested_frames,), dtype=np.int64)
    molecule_index_all = np.zeros((total_requested_frames,), dtype=np.int64)
    frame_idx_all = np.zeros((total_requested_frames,), dtype=np.int64)
    astate_all = np.ones((total_requested_frames,), dtype=np.int64)
    target_state_all = np.zeros((total_requested_frames,), dtype=np.int64)

    sample_idx = 0
    skipped_frames = []
    for mol_id, state_rows, n_keep in molecule_blocks:
        s0, s1, s2 = state_rows
        ground_spec = bond_specs[(mol_id, GROUND_STATE)]
        n_ground_bond_edges = int(ground_spec.bond_idx_i.shape[0])
        for local_t in range(n_keep):
            natoms = s0.natoms
            source_frame_idx = int(s0.frame_idx[local_t])
            r_padded = pad_atomwise(s0.R[local_t], max_atoms, 0.0)
            idx_i, idx_j, pair_mask = build_sample_pair_tensors(
                R=r_padded,
                natoms=natoms,
                base_i=base_i,
                base_j=base_j,
                r_cut=r_cut,
            )
            state_annotations = {}
            try:
                for state in (GROUND_STATE, DELTA_STATE_1, DELTA_STATE_2):
                    state_data = state_rows[state]
                    state_annotations[state] = build_bond_annotations(
                        pair_mask=pair_mask,
                        R=r_padded,
                        spec=bond_specs[(mol_id, state)],
                        pair_slot_lookup=pair_slot_lookup,
                        mol_id=mol_id,
                        state_label=state,
                        frame_idx=int(state_data.frame_idx[local_t]),
                        r_cut=r_cut,
                    )
            except MissingDirectedBondError as exc:
                skipped_frames.append(
                    {
                        "mol_id": mol_id,
                        "frame_idx": source_frame_idx,
                        "reason": str(exc),
                    }
                )
                warnings.warn(f"Skipping delta frame due to missing directed bond: {exc}")
                continue

            r_all[sample_idx] = r_padded
            f_all[sample_idx] = pad_atomwise(s0.F[local_t], max_atoms, 0.0)
            e_all[sample_idx] = s0.E[local_t]
            delta_e1_all[sample_idx] = s1.E[local_t] - s0.E[local_t]
            delta_e2_all[sample_idx] = s2.E[local_t] - s0.E[local_t]
            delta_f1_all[sample_idx] = pad_atomwise(
                s1.F[local_t] - s0.F[local_t], max_atoms, 0.0
            )
            delta_f2_all[sample_idx] = pad_atomwise(
                s2.F[local_t] - s0.F[local_t], max_atoms, 0.0
            )
            z_all[sample_idx] = pad_atomwise(s0.z_per_frame[local_t], max_atoms, 0)
            node_mask_all[sample_idx] = pad_atomwise(
                s0.node_mask[local_t], max_atoms, False
            )
            idx_i_all[sample_idx] = idx_i
            idx_j_all[sample_idx] = idx_j
            pair_mask_all[sample_idx] = pair_mask
            for state, (bond_prob, bond_mask) in state_annotations.items():
                bond_prob_by_state[state][sample_idx] = bond_prob
                bond_mask_by_state[state][sample_idx] = bond_mask
            bond_idx_i_all[sample_idx, :n_ground_bond_edges] = ground_spec.bond_idx_i
            bond_idx_j_all[sample_idx, :n_ground_bond_edges] = ground_spec.bond_idx_j
            n_atoms_all[sample_idx] = natoms
            molecule_index_all[sample_idx] = molecule_to_index[mol_id]
            frame_idx_all[sample_idx] = source_frame_idx
            sample_idx += 1

    save_dict = {
        "R": r_all[:sample_idx],
        "F": f_all[:sample_idx],
        "E": e_all[:sample_idx],
        "Delta_E1": delta_e1_all[:sample_idx],
        "Delta_E2": delta_e2_all[:sample_idx],
        "Delta_F1": delta_f1_all[:sample_idx],
        "Delta_F2": delta_f2_all[:sample_idx],
        "z": z_all[:sample_idx],
        "atom_state": atom_state_all[:sample_idx],
        "node_mask": node_mask_all[:sample_idx],
        "idx_i": idx_i_all[:sample_idx],
        "idx_j": idx_j_all[:sample_idx],
        "pair_mask": pair_mask_all[:sample_idx],
        "bond_idx_i": bond_idx_i_all[:sample_idx],
        "bond_idx_j": bond_idx_j_all[:sample_idx],
        "bond_prob": bond_prob_by_state[GROUND_STATE][:sample_idx],
        "bond_mask": bond_mask_by_state[GROUND_STATE][:sample_idx],
        "bond_prob_s0": bond_prob_by_state[GROUND_STATE][:sample_idx],
        "bond_mask_s0": bond_mask_by_state[GROUND_STATE][:sample_idx],
        "bond_prob_s1": bond_prob_by_state[DELTA_STATE_1][:sample_idx],
        "bond_mask_s1": bond_mask_by_state[DELTA_STATE_1][:sample_idx],
        "bond_prob_s2": bond_prob_by_state[DELTA_STATE_2][:sample_idx],
        "bond_mask_s2": bond_mask_by_state[DELTA_STATE_2][:sample_idx],
        "bond_reference_local_t": bond_reference_local_t_all[:sample_idx],
        "bond_reference_frame_idx": bond_reference_frame_idx_all[:sample_idx],
        "n_atoms": n_atoms_all[:sample_idx],
        "molecule_index": molecule_index_all[:sample_idx],
        "frame_idx": frame_idx_all[:sample_idx],
        "astate": astate_all[:sample_idx],
        "target_state": target_state_all[:sample_idx],
        "molecule_names": np.asarray(kept_molecules),
    }

    state_data = [state for _, state_rows, _ in molecule_blocks for state in state_rows]
    meta = {
        "source": "filtered_shnitsel_data_saad npy directories",
        "builder": "make_delta_dataset.py",
        "model_intent": (
            "delta model with S1-S0 and S2-S0 targets plus state-specific bonds"
        ),
        "units": {"R": "angstrom", "E": "eV", "F": "eV/angstrom"},
        "selection_logic": (
            "row-index paired filtered states; R/E/F come from state_1, "
            "Delta_E1/F1 from state_2 - state_1, Delta_E2/F2 from state_3 - state_1"
        ),
        "delta_pairing_assumption": (
            "The filtered state_1/state_2/state_3 rows must refer to the same "
            "geometries for deltas to be physically meaningful."
        ),
        "geometry_alignment_checked": bool(check_geometry_alignment),
        "geometry_atol": float(geometry_atol),
        "molecules": kept_molecules,
        "skipped_molecules": skipped_molecules,
        "numframes": None if numframes is None else int(numframes),
        "frames_per_molecule": (
            None if frames_per_molecule is None else int(frames_per_molecule)
        ),
        "requested_total_frames": int(total_requested_frames),
        "total_frames": int(sample_idx),
        "skipped_frame_count": int(len(skipped_frames)),
        "skipped_frames": skipped_frames,
        "max_atoms": int(max_atoms),
        "max_pairs": int(max_pairs),
        "max_bond_edges": int(max_bond_edges),
        "r_cut": float(r_cut),
        "bond_prob_channels": list(BOND_PROB_CHANNELS),
        "bond_state_keys": {
            "state_0": ["bond_prob_s0", "bond_mask_s0"],
            "state_1": ["bond_prob_s1", "bond_mask_s1"],
            "state_2": ["bond_prob_s2", "bond_mask_s2"],
        },
        "canonical_bond_keys": "bond_prob and bond_mask are aliases for state 0.",
        "canonical_bond_state": 0,
        "state_data": state_data_summary(state_data),
    }
    _apply_metadata_overrides(
        meta,
        metadata_overrides,
        protected_keys=(
            "geometry_alignment_checked",
            "geometry_atol",
            "molecules",
            "skipped_molecules",
            "numframes",
            "frames_per_molecule",
            "requested_total_frames",
            "total_frames",
            "skipped_frame_count",
            "skipped_frames",
            "max_atoms",
            "max_pairs",
            "max_bond_edges",
            "r_cut",
            "canonical_bond_state",
            "state_data",
        ),
    )
    write_npz_and_metadata(output, save_dict, meta)


def assemble_delta_offset_dataset(
    *,
    source: StateSource,
    output: str | Path,
    molecules: list[str] | None,
    states: list[int],
    numframes: int | None,
    r_cut: float,
    bond_specs_file: str | Path,
    skip_missing_bond_specs: bool,
    metadata_overrides: dict[str, Any] | None = None,
) -> None:
    normalized_states = validate_excited_states(states)
    if bond_specs_file is None:
        raise ValueError("bond_specs_file is required for delta-offset construction.")

    molecule_ids = source.discover_molecule_ids(molecules)
    loaded_state_data = _load_state_grid(
        source,
        molecule_ids,
        normalized_states,
        active_only=True,
    )
    validate_active_labels(loaded_state_data)
    loaded_state_data = trim_trailing_padding(loaded_state_data)

    empty_state_entries = [
        {
            "mol_id": data.mol_id,
            "active_state": int(data.state_label),
            "reason": "no source frames have this active-state label",
        }
        for data in loaded_state_data
        if data.nframes == 0
    ]
    if empty_state_entries:
        warnings.warn(
            "Skipping molecule/state entries with no matching active frames: "
            f"{empty_state_entries}",
            RuntimeWarning,
            stacklevel=2,
        )
        loaded_state_data = [data for data in loaded_state_data if data.nframes]
    if not loaded_state_data:
        raise ValueError(
            "No source frames match the requested delta-offset active states."
        )

    bond_specs = load_bond_specs(bond_specs_file)
    state_data, skipped_bond_state_entries = select_state_data_for_bonds(
        loaded_state_data,
        bond_specs,
        skip_missing_bond_specs=skip_missing_bond_specs,
    )
    skipped_state_entries = empty_state_entries + skipped_bond_state_entries
    if not state_data:
        raise ValueError("No molecule/state data left for delta-offset construction.")

    kept_molecules = [
        mol_id
        for mol_id in molecule_ids
        if any(data.mol_id == mol_id for data in state_data)
    ]
    state_data, frames_per_molecule = balance_state_data_by_molecule(
        state_data,
        kept_molecules,
        numframes,
    )
    molecule_to_index = {
        mol_id: index for index, mol_id in enumerate(kept_molecules)
    }
    total_requested_frames = sum(data.nframes for data in state_data)
    if total_requested_frames == 0:
        raise ValueError("Delta-offset construction selected zero active-state frames.")

    max_atoms = max(data.natoms for data in state_data)
    base_i, base_j, pair_slot_lookup, max_pairs = base_graph_context(max_atoms)

    r_all = np.zeros((total_requested_frames, max_atoms, 3), dtype=np.float32)
    f_all = np.zeros((total_requested_frames, max_atoms, 3), dtype=np.float32)
    e_all = np.zeros((total_requested_frames,), dtype=np.float32)
    z_all = np.zeros((total_requested_frames, max_atoms), dtype=np.int64)
    atom_state_all = np.zeros((total_requested_frames, max_atoms), dtype=np.int64)
    node_mask_all = np.zeros((total_requested_frames, max_atoms), dtype=bool)
    idx_i_all = np.full((total_requested_frames, max_pairs), -1, dtype=np.int64)
    idx_j_all = np.full((total_requested_frames, max_pairs), -1, dtype=np.int64)
    pair_mask_all = np.zeros((total_requested_frames, max_pairs), dtype=bool)
    n_atoms_all = np.zeros((total_requested_frames,), dtype=np.int64)
    molecule_index_all = np.zeros((total_requested_frames,), dtype=np.int64)
    frame_idx_all = np.zeros((total_requested_frames,), dtype=np.int64)
    astate_all = np.zeros((total_requested_frames,), dtype=np.int64)
    target_state_all = np.zeros((total_requested_frames,), dtype=np.int64)

    bond_prob_all = np.zeros(
        (total_requested_frames, max_pairs, len(BOND_PROB_CHANNELS)),
        dtype=np.float32,
    )
    bond_mask_all = np.zeros((total_requested_frames, max_pairs), dtype=bool)
    bond_prob_s0_all = np.zeros_like(bond_prob_all)
    bond_mask_s0_all = np.zeros_like(bond_mask_all)

    sample_idx = 0
    skipped_frames = []
    for data in state_data:
        active_state = int(data.state_label)
        for local_t in range(data.nframes):
            natoms = data.natoms
            source_frame_idx = int(data.frame_idx[local_t])
            r_padded = pad_atomwise(data.R[local_t], max_atoms, 0.0)
            idx_i, idx_j, pair_mask = build_sample_pair_tensors(
                R=r_padded,
                natoms=natoms,
                base_i=base_i,
                base_j=base_j,
                r_cut=r_cut,
            )

            try:
                active_bond_prob, active_bond_mask = build_bond_annotations(
                    pair_mask=pair_mask,
                    R=r_padded,
                    spec=bond_specs[(data.mol_id, active_state)],
                    pair_slot_lookup=pair_slot_lookup,
                    mol_id=data.mol_id,
                    state_label=active_state,
                    frame_idx=source_frame_idx,
                    r_cut=r_cut,
                )
                active_bond_prob, active_bond_mask = normalize_bond_annotations(
                    active_bond_prob,
                    active_bond_mask,
                    context=(
                        f"{data.mol_id} state {active_state} frame {source_frame_idx}"
                    ),
                )
                ground_bond_prob, ground_bond_mask = build_bond_annotations(
                    pair_mask=pair_mask,
                    R=r_padded,
                    spec=bond_specs[(data.mol_id, GROUND_STATE)],
                    pair_slot_lookup=pair_slot_lookup,
                    mol_id=data.mol_id,
                    state_label=GROUND_STATE,
                    frame_idx=source_frame_idx,
                    r_cut=r_cut,
                )
                ground_bond_prob, ground_bond_mask = normalize_bond_annotations(
                    ground_bond_prob,
                    ground_bond_mask,
                    context=(
                        f"{data.mol_id} state {GROUND_STATE} frame {source_frame_idx}"
                    ),
                )
            except MissingDirectedBondError as exc:
                skipped_frames.append(
                    {
                        "mol_id": data.mol_id,
                        "active_state": active_state,
                        "frame_idx": source_frame_idx,
                        "reason": str(exc),
                    }
                )
                warnings.warn(
                    f"Skipping delta-offset frame due to missing directed bond: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            r_all[sample_idx] = r_padded
            f_all[sample_idx] = pad_atomwise(data.F[local_t], max_atoms, 0.0)
            e_all[sample_idx] = data.E[local_t]
            z_all[sample_idx] = pad_atomwise(
                data.z_per_frame[local_t],
                max_atoms,
                0,
            )
            atom_state_all[sample_idx, :natoms] = active_state
            node_mask_all[sample_idx] = pad_atomwise(
                data.node_mask[local_t],
                max_atoms,
                False,
            )
            idx_i_all[sample_idx] = idx_i
            idx_j_all[sample_idx] = idx_j
            pair_mask_all[sample_idx] = pair_mask
            n_atoms_all[sample_idx] = natoms
            molecule_index_all[sample_idx] = molecule_to_index[data.mol_id]
            frame_idx_all[sample_idx] = source_frame_idx
            astate_all[sample_idx] = active_state
            target_state_all[sample_idx] = active_state

            bond_prob_all[sample_idx] = active_bond_prob
            bond_mask_all[sample_idx] = active_bond_mask
            bond_prob_s0_all[sample_idx] = ground_bond_prob
            bond_mask_s0_all[sample_idx] = ground_bond_mask
            sample_idx += 1

    if sample_idx == 0:
        raise ValueError(
            "Every selected delta-offset frame was rejected by bond/cutoff validation."
        )

    used_molecule_indices = sorted(
        set(int(index) for index in molecule_index_all[:sample_idx])
    )
    molecule_index_remap = {
        old_index: new_index
        for new_index, old_index in enumerate(used_molecule_indices)
    }
    emitted_molecule_index = np.asarray(
        [
            molecule_index_remap[int(index)]
            for index in molecule_index_all[:sample_idx]
        ],
        dtype=np.int64,
    )
    emitted_molecules = [kept_molecules[index] for index in used_molecule_indices]
    emitted_states = sorted(set(int(state) for state in astate_all[:sample_idx]))

    save_dict = {
        "R": r_all[:sample_idx],
        "F": f_all[:sample_idx],
        "E": e_all[:sample_idx],
        "z": z_all[:sample_idx],
        "atom_state": atom_state_all[:sample_idx],
        "node_mask": node_mask_all[:sample_idx],
        "idx_i": idx_i_all[:sample_idx],
        "idx_j": idx_j_all[:sample_idx],
        "pair_mask": pair_mask_all[:sample_idx],
        "n_atoms": n_atoms_all[:sample_idx],
        "molecule_index": emitted_molecule_index,
        "frame_idx": frame_idx_all[:sample_idx],
        "astate": astate_all[:sample_idx],
        "target_state": target_state_all[:sample_idx],
        "molecule_names": np.asarray(emitted_molecules),
        "bond_prob": bond_prob_all[:sample_idx],
        "bond_mask": bond_mask_all[:sample_idx],
        "bond_prob_s0": bond_prob_s0_all[:sample_idx],
        "bond_mask_s0": bond_mask_s0_all[:sample_idx],
    }

    meta = {
        "source": "filtered_shnitsel_data_saad npy directories",
        "builder": "make_delta_offset_dataset.py",
        "model_intent": (
            "bond-aware delta-offset learning from active-state E/F labels"
        ),
        "training_mode": "delta_offset",
        "units": {"R": "angstrom", "E": "eV", "F": "eV/angstrom"},
        "selection_logic": (
            "independent state-expanded S1/S2 rows; E/F are the valid labels for "
            "each row's zero-based active state; no cross-state row pairing"
        ),
        "target_construction": (
            "Offset_E/F are intentionally absent and must be generated by the MLFF "
            "trainer as active E/F minus the pinned frozen S0 teacher prediction."
        ),
        "invalid_arrays": (
            "energy_all and forces_all are neither read nor written; Delta_E/F are "
            "not physical targets for this dataset."
        ),
        "state_indexing": {
            "astate": "zero-based active state; allowed values are 1 and/or 2",
            "target_state": "exact zero-based alias of astate",
            "atom_state": "zero-based active state on real atoms; padded atoms are 0",
        },
        "states": emitted_states,
        "requested_states": [int(state) for state in normalized_states],
        "molecules": emitted_molecules,
        "requested_molecules": molecule_ids,
        "numframes": None if numframes is None else int(numframes),
        "frames_per_molecule": (
            None if frames_per_molecule is None else int(frames_per_molecule)
        ),
        "requested_total_frames": int(total_requested_frames),
        "total_frames": int(sample_idx),
        "skipped_state_entry_count": int(len(skipped_state_entries)),
        "skipped_state_entries": skipped_state_entries,
        "skipped_frame_count": int(len(skipped_frames)),
        "skipped_frames": skipped_frames,
        "max_atoms": int(max_atoms),
        "max_pairs": int(max_pairs),
        "r_cut": float(r_cut),
        "bond_probability_tolerance": 1e-5,
        "bond_prob_channels": list(BOND_PROB_CHANNELS),
        "canonical_bond_keys": (
            "bond_prob and bond_mask describe each row's active state."
        ),
        "canonical_bond_state": "active_state",
        "ground_teacher_bond_keys": (
            "bond_prob_s0 and bond_mask_s0 describe S0 bonds on the same active "
            "geometry for a bond-aware frozen ground-state teacher."
        ),
        "state_data": state_data_summary(state_data),
    }

    _apply_metadata_overrides(
        meta,
        metadata_overrides,
        protected_keys=(
            "states",
            "requested_states",
            "molecules",
            "requested_molecules",
            "numframes",
            "frames_per_molecule",
            "requested_total_frames",
            "total_frames",
            "skipped_state_entry_count",
            "skipped_state_entries",
            "skipped_frame_count",
            "skipped_frames",
            "max_atoms",
            "max_pairs",
            "r_cut",
            "canonical_bond_state",
            "state_data",
        ),
    )
    write_npz_and_metadata(output, save_dict, meta)
