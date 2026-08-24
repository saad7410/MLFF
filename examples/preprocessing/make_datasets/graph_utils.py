from __future__ import annotations

import numpy as np


def all_directed_nonself_pairs(max_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed directed edge ordering over ``[0, max_atoms)`` without self edges."""
    idx = np.arange(max_atoms, dtype=np.int64)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    keep = ii != jj
    base_i = ii[keep].reshape(-1)
    base_j = jj[keep].reshape(-1)
    return base_i, base_j


def build_sample_pair_tensors(
    R: np.ndarray,
    natoms: int,
    base_i: np.ndarray,
    base_j: np.ndarray,
    r_cut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build directed cutoff edges for one padded structure using a fixed pair ordering."""
    if R.ndim != 2 or R.shape[-1] != 3:
        raise ValueError(f"Expected R with shape (max_atoms, 3), got {R.shape}")

    real_pair = (base_i < natoms) & (base_j < natoms)
    dR = R[base_j] - R[base_i]
    distances = np.linalg.norm(dR, axis=-1)
    valid = real_pair & (distances <= float(r_cut)) & (distances > 0.0)

    idx_i = np.where(valid, base_i, -1).astype(np.int64)
    idx_j = np.where(valid, base_j, -1).astype(np.int64)
    pair_mask = valid.astype(bool)
    return idx_i, idx_j, pair_mask
