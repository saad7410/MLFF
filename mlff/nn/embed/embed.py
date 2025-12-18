"""
embed.py (SO3krates / MLFF)

This file defines "embedding" modules that convert raw molecular inputs into
feature tensors used by the rest of the network.

Two big categories:
  1) Geometry-based embeddings (from positions + neighbor list):
     - distances, radial basis expansions, cutoff weights,
     - spherical harmonics / solid harmonics,
     - optional per-atom "spherical harmonic coordinates" (SPHCs, chi).

  2) Atom-type embeddings (from atomic numbers):
     - learned embedding lookup (nn.Embed) OR
     - one-hot encoding.

Key idea: the model operates on a neighbor graph.
  - idx_i[k], idx_j[k] define the k-th edge (i <- j) i.e. "j is a neighbor of i"
  - pair_mask marks which edges are real vs. padding
  - point_mask marks which atoms are real vs. padding
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import logging

from functools import partial
from typing import Any, Dict, Sequence
from jax.ops import segment_sum

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_mask
from mlff.masking.mask import safe_scale
from mlff.basis_function.radial import get_rbf_fn
from mlff.cutoff_function import add_cell_offsets
from mlff.cutoff_function import get_cutoff_fn
from mlff.basis_function.spherical import init_sph_fn
from mlff.properties import property_names as pn


# TODO: write init_from_dict methods in order to improve backward compatibility. E.g. AtomTypeEmbed(**h)
# will only work as long as the non-default properties of the class are exactly the ones equal to the ones in h. As soon
# as additional arguments appear in h. Maybe use something like kwargs to allow for extensions?

class GeometryEmbed(BaseSubModule):
    """
    Build geometric features on *edges* (pairs) and optionally on *nodes* (atoms).

    Inputs expected in `__call__` (for one structure/frame):
      - positions (or displacements) and neighbor graph indices
      - masks for padded atoms and padded edges

    Main outputs:
      - r_ij      : displacement vectors for each pair k, shape (n_pairs, 3)
      - d_ij      : distances for each pair, shape (n_pairs,)
      - rbf_ij    : radial basis expansion of d_ij, shape (n_pairs, K)
      - phi_r_cut : cutoff weights, shape (n_pairs,)
      - sph_ij    : spherical harmonics of unit vectors, shape (n_pairs, m_tot)
      - chi       : (optional) per-atom SPHC embedding, shape (n_atoms, m_tot)
      - g_ij      : (optional) solid harmonics, shape (n_pairs, m_tot, K)

    Parameters (passed from config):
      - degrees: list of spherical harmonic degrees ℓ (e.g. [0,1,2,3])
      - n_rbf:   number of radial basis functions (K)
      - r_cut:   cutoff radius for neighbor graph and cutoff function
      - sphc:    whether to compute per-atom spherical harmonic coordinates chi
      - mic:     whether to use periodic boundary corrections (minimal image convention)
      - input_convention:
          'positions'     -> compute r_ij = R[j] - R[i]
          'displacements' -> r_ij is provided directly in the input dict
    """

    # ---- configuration fields (Flax-style dataclass fields) ----
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_function: str
    n_rbf: int
    radial_cutoff_fn: str
    r_cut: float
    sphc: bool
    sphc_normalization: float = None
    mic: bool = False
    solid_harmonic: bool = False
    input_convention: str = "positions"
    module_name: str = "geometry_embed"

    def setup(self):
        """
        Called once when the module is initialized.
        We:
          - decide which input keys to use (positions vs displacements)
          - create callable functions for spherical harmonics, RBF, and cutoff
        """
        # ---- decide how to get geometry information from inputs ----
        if self.input_convention == "positions":
            # atomic positions R are provided in the input dict
            self.atomic_position_key = self.prop_keys.get("atomic_position")

            # mic argument compatibility warnings
            if self.mic == "bins":
                logging.warning(f"mic={self.mic} is deprecated in favor of mic=True.")
            if self.mic == "naive":
                raise DeprecationWarning(f"mic={self.mic} is not longer supported.")

            # if periodic boundary conditions are used, we also need unit cell and
            # per-edge cell offsets
            if self.mic:
                self.unit_cell_key = self.prop_keys.get("unit_cell")
                self.cell_offset_key = self.prop_keys.get("cell_offset")

        elif self.input_convention == "displacements":
            # displacement vectors r_ij are provided directly in the input dict
            self.displacement_vector_key = self.prop_keys.get("displacement_vector")

        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Always needed: atomic types z (mostly for masks / optional chi init)
        self.atomic_type_key = self.prop_keys.get("atomic_type")

        # ---- spherical harmonics functions for each degree ℓ ----
        # Each init_sph_fn(ℓ) returns a callable: (n_pairs, 3) -> (n_pairs, 2ℓ+1)
        self.sph_fns = [init_sph_fn(l) for l in self.degrees]

        # ---- radial basis function for distances ----
        # get_rbf_fn returns a factory; we instantiate with (n_rbf, r_cut)
        _rbf_fn = get_rbf_fn(self.radial_basis_function)
        self.rbf_fn = _rbf_fn(n_rbf=self.n_rbf, r_cut=self.r_cut)

        # ---- smooth cutoff function ----
        # get_cutoff_fn returns a function; we partial in r_cut
        _cut_fn = get_cutoff_fn(self.radial_cutoff_fn)
        self.cut_fn = partial(_cut_fn, r_cut=self.r_cut)

        # Store normalization constant for chi (SPHC embedding)
        self._lambda = jnp.float32(self.sphc_normalization)

    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        Compute geometry embeddings for a single structure/frame.

        Required keys in `inputs`:
          - idx_i: (n_pairs,) central/receiver atom indices i
          - idx_j: (n_pairs,) neighbor/sender atom indices j
          - pair_mask: (n_pairs,) 1 for real edges, 0 for padded edges
          - point_mask: (n_atoms,) 1 for real atoms, 0 for padded atoms (used if sphc)
          - atomic positions (if input_convention == 'positions'):
              - atomic_position_key -> R: (n_atoms, 3)
              - unit_cell_key: (3, 3) and cell_offset_key: (n_pairs, 3) if mic=True
          - OR displacement vectors (if input_convention == 'displacements'):
              - displacement_vector_key -> r_ij: (n_pairs, 3)

        Returns:
          A dict of geometric tensors (see class docstring).
        """
        # ---- neighbor graph edge list (compressed adjacency) ----
        idx_i = inputs["idx_i"]          # (n_pairs,) receiver atom index per edge
        idx_j = inputs["idx_j"]          # (n_pairs,) sender   atom index per edge
        pair_mask = inputs["pair_mask"]  # (n_pairs,) 1/0 mask for real vs padded edges

        # ---- compute displacement vectors r_ij for each edge (i <- j) ----
        if self.input_convention == "positions":
            # R: (n_atoms, 3)
            R = inputs[self.atomic_position_key]

            # Vectorized: r_ij[k] = R[idx_j[k]] - R[idx_i[k]]
            # safe_scale applies pair_mask: padded edges become 0 vectors
            r_ij = safe_scale(
                jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j),
                scale=pair_mask[:, None]
            )  # (n_pairs, 3)

            # If periodic boundary conditions (minimal image convention) are enabled:
            # Add cell offsets so that r_ij uses the correct periodic image.
            if self.mic:
                cell = inputs[self.unit_cell_key]               # (3, 3)
                cell_offsets = inputs[self.cell_offset_key]     # (n_pairs, 3)
                r_ij = add_cell_offsets(
                    r_ij=r_ij,
                    cell=cell,
                    cell_offsets=cell_offsets
                )  # (n_pairs, 3)

        elif self.input_convention == "displacements":
            # Some pipelines precompute r_ij externally and pass it in.
            R = None
            r_ij = inputs[self.displacement_vector_key]         # (n_pairs, 3)

        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Make sure padded edges contribute nothing
        r_ij = safe_scale(r_ij, scale=pair_mask[:, None])        # (n_pairs, 3)

        # ---- scalar distances d_ij = ||r_ij|| ----
        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # (n_pairs,)

        # ---- radial basis expansion (RBF) of distances ----
        # self.rbf_fn expects shape (n_pairs, 1) typically; outputs (n_pairs, K)
        rbf_ij = safe_scale(self.rbf_fn(d_ij[:, None]), scale=pair_mask[:, None])  # (n_pairs, K)

        # ---- cutoff weights phi(r) in [0,1], goes to 0 near r_cut ----
        phi_r_cut = safe_scale(self.cut_fn(d_ij), scale=pair_mask)  # (n_pairs,)

        # ---- normalized direction vectors r̂_ij (avoid divide by zero) ----
        unit_r_ij = safe_mask(
            mask=d_ij[:, None] != 0,     # only normalize non-zero distances
            operand=r_ij,
            fn=lambda y: y / d_ij[:, None],
            placeholder=0
        )  # (n_pairs, 3)
        unit_r_ij = safe_scale(unit_r_ij, scale=pair_mask[:, None])  # (n_pairs, 3)

        # ---- spherical harmonics Y_{ℓm}(r̂_ij) for each requested degree ℓ ----
        sph_harms_ij = []
        for sph_fn in self.sph_fns:
            # sph_fn(unit_r_ij) -> (n_pairs, 2ℓ+1)
            sph_ij = safe_scale(sph_fn(unit_r_ij), scale=pair_mask[:, None])
            sph_harms_ij.append(sph_ij)

        # Concatenate all ℓ blocks into one feature vector per edge:
        # m_tot = Σ_{ℓ in degrees} (2ℓ+1)
        sph_harms_ij = jnp.concatenate(sph_harms_ij, axis=-1) if len(self.degrees) > 0 else None
        # sph_harms_ij: (n_pairs, m_tot)

        # ---- package the geometric tensors ----
        geometric_data = {
            "R": R,                   # (n_atoms, 3) or None if using displacements
            "r_ij": r_ij,             # (n_pairs, 3)
            "unit_r_ij": unit_r_ij,   # (n_pairs, 3)
            "d_ij": d_ij,             # (n_pairs,)
            "rbf_ij": rbf_ij,         # (n_pairs, K)
            "phi_r_cut": phi_r_cut,   # (n_pairs,)
            "sph_ij": sph_harms_ij,   # (n_pairs, m_tot)
        }

        # ---- optional: per-atom spherical harmonic coordinates (SPHCs) chi ----
        # chi is a node-level feature computed by summing edge spherical features
        # into each central atom i using segment_sum over idx_i.
        if self.sphc:
            z = inputs[self.atomic_type_key]        # (n_atoms,)
            point_mask = inputs["point_mask"]       # (n_atoms,)

            # NOTE: In the upstream code the naming is slightly confusing:
            # - The branch checks sphc_normalization is None
            # - and then calls _init_sphc_zeros
            #
            # In practice:
            # - _init_sphc_zeros -> chi = 0
            # - _init_sphc       -> chi computed from neighborhood
            #
            # Keep logic identical to upstream for reproducibility.
            if self.sphc_normalization is None:
                # Initialize chi to zeros (no neighborhood-based pre-embedding)
                geometric_data.update(
                    _init_sphc_zeros(
                        z=z,
                        sph_ij=sph_harms_ij,
                        phi_r_cut=phi_r_cut,
                        idx_i=idx_i,
                        point_mask=point_mask,
                        mp_normalization=self._lambda,
                    )
                )
            else:
                # Initialize chi using neighborhood spherical harmonics (weighted by cutoff)
                geometric_data.update(
                    _init_sphc(
                        z=z,
                        sph_ij=sph_harms_ij,
                        phi_r_cut=phi_r_cut,
                        idx_i=idx_i,
                        point_mask=point_mask,
                        mp_normalization=self._lambda,
                    )
                )

        # ---- optional: solid harmonics g_ij = Y(r̂_ij) ⊗ RBF(d_ij) ----
        # This is a richer edge representation combining angular + radial info.
        if self.solid_harmonic:
            # Weight radial features by cutoff
            rbf_ij_cut = safe_scale(rbf_ij, scale=phi_r_cut[:, None])  # (n_pairs, K)

            # Outer-product style combination:
            # sph_harms_ij: (n_pairs, m_tot)
            # rbf_ij_cut :  (n_pairs, K)
            # g_ij:         (n_pairs, m_tot, K)
            g_ij = sph_harms_ij[:, :, None] * rbf_ij_cut[:, None, :]

            # Mask padded edges
            g_ij = safe_scale(g_ij, scale=pair_mask[:, None, None], placeholder=0)

            geometric_data.update({"g_ij": g_ij})

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        """Utility to change how geometry is read (positions vs displacements)."""
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        """Metadata export for logging / saving hyperparameters."""
        return {
            self.module_name: {
                "degrees": self.degrees,
                "radial_basis_function": self.radial_basis_function,
                "n_rbf": self.n_rbf,
                "radial_cutoff_fn": self.radial_cutoff_fn,
                "r_cut": self.r_cut,
                "sphc": self.sphc,
                "sphc_normalization": self.sphc_normalization,
                "solid_harmonic": self.solid_harmonic,
                "mic": self.mic,
                "input_convention": self.input_convention,
                "prop_keys": self.prop_keys,
            }
        }


def _init_sphc(z, sph_ij, phi_r_cut, idx_i, point_mask, mp_normalization, *args, **kwargs):
    """
    Compute per-atom SPHC features chi by summing neighbor spherical harmonics.

    Steps:
      1) weight edge spherical harmonics by cutoff phi(r)
      2) segment_sum over idx_i -> aggregate all edge features into each atom i
      3) apply point_mask to zero padded atoms
      4) normalize by mp_normalization (lambda)

    Shapes:
      sph_ij:      (n_pairs, m_tot)
      phi_r_cut:   (n_pairs,)
      idx_i:       (n_pairs,)
      point_mask:  (n_atoms,)
      chi:         (n_atoms, m_tot)
    """
    # Apply cutoff weighting to the spherical features
    _sph_harms_ij = safe_scale(sph_ij, phi_r_cut[:, None])  # (n_pairs, m_tot)

    # Sum per receiver atom i (central atom)
    chi = segment_sum(_sph_harms_ij, segment_ids=idx_i, num_segments=len(z))  # (n_atoms, m_tot)

    # Mask out padded atoms
    chi = safe_scale(chi, scale=point_mask[:, None])  # (n_atoms, m_tot)

    # Normalize (helps stabilize magnitude across different neighbor counts)
    return {"chi": chi / mp_normalization}


def _init_sphc_zeros(z, sph_ij, *args, **kwargs):
    """
    Initialize chi as all zeros (no neighborhood information at initialization time).

    Shape:
      chi: (n_atoms, m_tot)
    """
    return {"chi": jnp.zeros((z.shape[-1], sph_ij.shape[-1]), dtype=sph_ij.dtype)}


class AtomTypeEmbed(BaseSubModule):
    """
    Learned embedding for atom types.

    This is the same idea as word embeddings in NLP:
      - you have an integer token (atomic number z)
      - you look up a trainable vector of length `features`

    Inputs:
      - z: (n_atoms,)
      - point_mask: (n_atoms,)

    Output:
      - embeddings: (n_atoms, features)
    """

    features: int
    prop_keys: Dict
    num_embeddings: int = 100
    module_name: str = "atom_type_embed"

    def setup(self):
        self.atomic_type_key = self.prop_keys.get("atomic_type")

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Args:
            inputs:
              - z: atomic types (n_atoms,)
              - point_mask: (n_atoms,) 1 for real atoms, 0 for padded atoms

        Returns:
            Atomic type embeddings, shape: (n_atoms, features)
        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs["point_mask"]

        # Ensure integer indexing into embedding table
        z = z.astype(jnp.int32)

        # Embedding lookup: table[num_embeddings, features], output (n_atoms, features)
        emb = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z)

        # Zero out embeddings for padded atoms
        return safe_scale(emb, scale=point_mask[:, None])

    def __dict_repr__(self):
        return {
            self.module_name: {
                "num_embeddings": self.num_embeddings,
                "features": self.features,
                "prop_keys": self.prop_keys,
            }
        }


class OneHotEmbed(BaseSubModule):
    """
    One-hot encoding for atom types (non-learned alternative to AtomTypeEmbed).

    Inputs:
      - z: (n_atoms,)

    Output:
      - z_one_hot: (n_atoms, n_types)
    """

    prop_keys: Dict
    atomic_types: Sequence[int]
    module_name: str = "one_hot_embed"

    def setup(self):
        # Create a callable that maps atomic numbers -> one-hot vectors
        self.to_one_hot = lambda x: to_onehot(x, node_types=self.atomic_types)
        self.atomic_type_key = self.prop_keys[pn.atomic_type]

    def __call__(self, inputs: Dict, *args, **kwargs):
        z = inputs[self.atomic_type_key]
        return {"z_one_hot": self.to_one_hot(z)}

    def __dict_repr__(self):
        return {
            self.module_name: {
                "atomic_types": self.atomic_types,
                "prop_keys": self.prop_keys,
            }
        }

    def reset_input_convention(self, input_convention: str) -> None:
        # Included for interface compatibility; not used here.
        pass


def to_onehot(features: jnp.ndarray, node_types: Sequence):
    r"""
    Create one-hot encoded vectors from integer node labels.

    Args:
        features:
            Array of node labels (atomic numbers), shape: (n_atoms,)
        node_types:
            List/sequence of allowed node types, e.g. [1, 6, 7, 8]

    Returns:
        One-hot matrix, shape: (n_atoms, len(node_types))

    Example:
        features = [6, 6, 8, 1]
        node_types = [1, 6, 8]
        -> [
             [0,1,0],
             [0,1,0],
             [0,0,1],
             [1,0,0]
           ]
    """
    cols = []
    for e in node_types:
        # For each allowed type e, build a column:
        # 1 where features == e else 0
        col = jnp.where(features == e, jnp.ones(1), jnp.zeros(1))[..., None]  # (n_atoms, 1)
        cols.append(col)

    return jnp.concatenate(cols, axis=-1)  # (n_atoms, n_types)
