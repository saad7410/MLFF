import jax
import numpy as np
import logging
import os

from functools import partial
from flax.traverse_util import flatten_dict, unflatten_dict
from pathlib import Path

from typing import Dict, Sequence

from mlff.indexing.indices import get_indices, get_pbc_neighbors
from mlff.random.random import set_seeds
from mlff.io import read_json, save_dict, merge_dicts
from mlff.data.preprocessing import get_per_atom_shift
from mlff.properties import property_names as pn


# Fix the public descriptor order so model inputs and dataset metadata cannot silently disagree.
BOND_CHANNELS = ('single', 'aromatic', 'double', 'triple')


def load_precomputed_graph_metadata(npz_path, r_cut):
    # Accept the conventional sibling JSON name and the explicit `.npz.json` variant.
    npz_path = Path(npz_path)
    candidates = (npz_path.with_suffix('.json'), Path(f'{npz_path}.json'))
    metadata_path = next((path for path in candidates if path.exists()), None)

    if metadata_path is None:
        # Missing sidecars are allowed for compatibility, but users lose cutoff/channel provenance checks.
        logging.warning('No adjacent graph metadata JSON found for %s; validating NPZ arrays only.', npz_path)
        return None

    # Load the sidecar once and validate every explicit compatibility field.
    metadata = read_json(metadata_path)
    validate_precomputed_graph_metadata(metadata=metadata, r_cut=r_cut, source=metadata_path)
    return metadata


def validate_precomputed_graph_metadata(metadata, r_cut, source='graph metadata'):
    # Reject an explicit cutoff mismatch because the supplied graph cannot be rebuilt safely.
    if 'r_cut' in metadata and not np.isclose(float(metadata['r_cut']), float(r_cut)):
        raise ValueError(f'Precomputed graph cutoff mismatch in {source}: '
                         f'{metadata["r_cut"]} != model r_cut {r_cut}.')

    # Support the source-pipeline names seen in adjacent dataset metadata.
    channel_keys = ('bond_channels', 'bond_prob_channels', 'bond_channel_order')
    channel_key = next((key for key in channel_keys if key in metadata), None)
    if channel_key is not None and tuple(metadata[channel_key]) != BOND_CHANNELS:
        raise ValueError(f'Bond channel metadata in {source} must be ordered as {BOND_CHANNELS}.')

    if 'r_cut' not in metadata or channel_key is None:
        # Incomplete metadata remains usable, while making the missing provenance visible.
        logging.warning('Graph metadata %s does not contain both `r_cut` and bond channel order.', source)

try:
    from jax import tree as _jtree
    tree_map = _jtree.map          # JAX ≥ 0.4.25 (incl. 0.6.x)
except Exception:
    from jax.tree_util import tree_map


class DataSet:
    def __init__(self, prop_keys: Dict, data: Dict, graph_metadata: Dict = None):
        self.prop_keys = prop_keys
        self.data = data
        self.graph_metadata = graph_metadata
        self.data, self.n_data = self._correct_shapes()
        self.splits = {}
        self.splits_info = {}
        self.data_split = {}
        self.scales = self._init_scales()

        self.track_shift_x_by_mean_x = []
        self.track_divide_x_by_std_y = []
        self.track_shift_x_by_type = []

    def summary(self):
        raise NotImplementedError

    def _correct_shapes(self):
        # get n_data, kind of messy but best way I could think of
        q_data = {k: v for k, v in self.data.items()}

        # Keep every structure-level energy target in the shared (B, 1) convention.
        scalar_energy_properties = (pn.energy, pn.delta_energy_1, pn.delta_energy_2)
        for property_name in scalar_energy_properties:
            property_key = self.prop_keys.get(property_name)
            if property_key in q_data:
                q_data[property_key] = q_data[property_key].reshape(-1, 1)

        def reshape(y):
            if len(y.shape) <= 1:
                return y.reshape(1, -1)
            else:
                return y

        q_data = tree_map(lambda y: reshape(y), q_data)
        max_key = max(tree_map(lambda y: len(y), q_data), key=tree_map(lambda y: len(y), q_data).get)
        n_data = len(q_data[max_key])

        def repeat(name, y, repeats):
            if len(y) == 1:
                print('Detected missing data dimension (0-th axis) for {}. Assume that the '
                      'data dimension is missing and repeat the entry {} times. Reshaped '
                      'array to ({}, {})'.format(name, n_data, n_data, y.shape[0]))
                return np.repeat(y, repeats=repeats, axis=0)
            elif 1 < len(y) < n_data:
                print('Detected missing data dimension (0-th axis) for {}. Assume that the '
                      'data dimension is missing and repeat the entry {} times. Reshaped '
                      'array to ({}, {})'.format(name, n_data, n_data, y.shape))
                return np.repeat(y[None], repeats=repeats, axis=0)
            else:
                return y

        return {k: repeat(name=k, y=v, repeats=n_data) for (k, v) in q_data.items()}, n_data

    def _init_scales(self):
        return {k: {'per_atom_shift': [0.] * 100,
                    'scale': 1.}
                for k in self.prop_keys.keys()}

    def save_scales(self, path, filename='scales.json'):
        save_dict(path=path, filename=filename, data=self.scales, exists_ok=True)

    def neighborhood_list(self, r_cut):
        R_key = self.prop_keys[pn.atomic_position]
        z_key = self.prop_keys[pn.atomic_type]
        neigh_idx = get_indices(R=self.data[R_key], z=self.data[z_key], r_cut=r_cut)
        return neigh_idx

    def average_number_of_neighbors(self) -> float:
        _d = self.data_split['train']
        _n_atoms = self.number_of_atoms()
        idx_i = _d['idx_i']  # shape: (n_data,P)
        p_segment_sum = partial(jax.ops.segment_sum, num_segments=_n_atoms)
        neighs = jax.vmap(p_segment_sum)(np.ones_like(idx_i), segment_ids=idx_i)  # shape: (n)
        return neighs.mean().item()

    def number_of_atoms(self) -> int:
        _n_atoms = (self.data_split['train'][self.prop_keys[pn.atomic_type]] != -1).sum(-1)  # shape: (D)
        if _n_atoms.std() != 0:
            logging.warning('Dataset contains structures with different structure sizes. Number of atoms is thus not'
                            'well defined.')
            return _n_atoms.item()
        else:
            return int(_n_atoms.mean().item())

    def all_atomic_types(self) -> Sequence[int]:
        z_key = self.prop_keys['atomic_type']
        z_unique = np.unique(self.data[z_key])
        return z_unique[z_unique > 0].tolist()

    def index_split(self,
                    data_idx_train,
                    data_idx_valid,
                    data_idx_test,
                    training: bool,
                    r_cut: float = None,
                    mic: str = None,
                    split_name: str = 'split',
                    precomputed_graph: bool = False):

        if mic == 'bins':
            logging.warning(f'mic={mic} is deprecated in favor of mic=True.')
        if mic == 'naive':
            raise DeprecationWarning(f'mic={mic} is not longer supported.')

        if precomputed_graph:
            # This first bond-aware port intentionally accepts only nonperiodic precomputed graphs.
            if mic:
                raise ValueError('Precomputed bond-aware graphs are currently supported only for nonperiodic data.')
            if r_cut is None:
                raise ValueError('`r_cut` is required to validate and record a precomputed graph split.')

            # Validate the full graph before slicing so every sample follows one aligned contract.
            self._validate_precomputed_graph(r_cut=r_cut)

        node_mask_present = False
        d = {}

        for i, i_n in zip([data_idx_train, data_idx_valid, data_idx_test], ['train', 'valid', 'test']):
            _d = {}
            for k, v in self.data.items():

                if k == self.prop_keys.get(pn.node_mask):
                    node_mask_present = True

                if len(v.shape) <= 1:
                    logging.warning(
                        f'Array with shape {v.shape} for quantity {k} detected, such that we assume that the data '
                        f'dimension is missing. Reshaped to ({len(i)}, {v.shape[-1]}).'
                    )
                    v = np.repeat(v[None, :], repeats=len(i), axis=0)
                _d.update({k: v[i]})

                d.update({i_n: _d})

        z_key = self.prop_keys.get(pn.atomic_type)
        node_msk_needed = node_mask_required(self.data[z_key])

        if precomputed_graph:
            # Preserve every supplied edge array and synthesize only the standard node mask when absent.
            if not node_mask_present:
                n_msk_key = self.prop_keys.get(pn.node_mask)
                for split in ('train', 'valid', 'test'):
                    d[split][n_msk_key] = (d[split][z_key] != 0)

            # Record reproducibility metadata without rebuilding or reordering the supplied graph.
            self._record_split(data_idx_train=data_idx_train,
                               data_idx_valid=data_idx_valid,
                               data_idx_test=data_idx_test,
                               split_name=split_name,
                               r_cut=r_cut,
                               mic=mic,
                               precomputed_graph=True)
            self.data_split = d
            return

        if r_cut is not None:
            R_key = self.prop_keys.get(pn.atomic_position)
            n_msk_key = self.prop_keys.get(pn.node_mask)

            if training:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key]])

                if node_msk_needed | node_mask_present:
                    n_msk_dat = np.concatenate([d['train'][n_msk_key], d['valid'][n_msk_key]])
                else:
                    n_msk_dat = np.ones_like(z_dat).astype(bool)

            else:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key], d['test'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key], d['test'][z_key]])

                if node_msk_needed | node_mask_present:
                    n_msk_dat = np.concatenate([d['train'][n_msk_key], d['valid'][n_msk_key], d['test'][n_msk_key]])
                else:
                    n_msk_dat = np.ones_like(z_dat).astype(bool)

            if mic:
                uc_key = self.prop_keys.get(pn.unit_cell)
                pbc_key = self.prop_keys.get(pn.pbc)

                unit_cell_dat = np.concatenate([d['train'][uc_key], d['valid'][uc_key], d['test'][uc_key]])
                pbc_dat = np.concatenate([d['train'][pbc_key], d['valid'][pbc_key], d['test'][pbc_key]])

                cell_lengths = np.linalg.norm(unit_cell_dat, axis=-1).reshape(-1)
                if 0.5 * min(cell_lengths) <= r_cut:
                    # raise NotImplementedError(f'Minimal image convention currently only implemented for '
                    #                           f'r_cut < 0.5*min(cell_lengths), but r_cut={r_cut} and '
                    #                           f'0.5*min(cell_lengths) = {0.5 * min(cell_lengths)}. Consider '
                    #                           f'using `get_pbc_indices` which uses ASE under the hood. '
                    #                           f'However, the latter takes ~15 times longer so maybe '
                    #                           f'reduce r_cut.')
                    logging.warning(f'r_cut > 0.5*min(cell_lengths) detected! r_cut={r_cut} and '
                                    f'0.5*min(cell_lengths) = {0.5 * min(cell_lengths)}. '
                                    f'This case has not been tested rigorously yet and might leave to unwanted'
                                    f'artifacts, so use with care!')

                neigh_idxs = get_pbc_neighbors(pos=R_dat,
                                               node_mask=n_msk_dat,
                                               cell=unit_cell_dat,
                                               cutoff=r_cut,
                                               pbc=pbc_dat)
            else:
                neigh_idxs = get_indices(R_dat, z_dat, r_cut=r_cut)

            n_train = len(data_idx_train)
            n_valid = len(data_idx_valid)

            idx_i_train, idx_i_valid, idx_i_test = np.split(neigh_idxs['idx_i'],
                                                            indices_or_sections=[n_train, n_train + n_valid])
            idx_j_train, idx_j_valid, idx_j_test = np.split(neigh_idxs['idx_j'],
                                                            indices_or_sections=[n_train, n_train + n_valid])

            if mic:
                c_off_train, c_off_valid, c_off_test = np.split(neigh_idxs['shifts'],
                                                                indices_or_sections=[n_train, n_train + n_valid])
                d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train, 'cell_offset': c_off_train},
                         'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid, 'cell_offset': c_off_valid},
                         'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test, 'cell_offset': c_off_test}}
            else:
                d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train},
                         'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid},
                         'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test}}

            _d = {k: merge_dicts(v, d_idx[k])
                  for k, v in d.items() if k in ['train', 'valid', 'test']}
            d.update(_d)

            if node_mask_present:
                pass
            else:
                n_msk_train, n_msk_valid, n_msk_test = np.split(n_msk_dat,
                                                                indices_or_sections=[n_train, n_train + n_valid])

                d_n_msk = {'train': {'node_mask': n_msk_train},
                           'valid': {'node_mask': n_msk_valid},
                           'test': {'node_mask': n_msk_test}}

                _d_n_msk = {k: merge_dicts(v, d_n_msk[k])
                            for k, v in d.items() if k in ['train', 'valid', 'test']}

                d.update(_d_n_msk)

        n_train = len(data_idx_train)
        n_valid = len(data_idx_valid)
        n_test = len(data_idx_test)
        self.splits.update({split_name: {'data_idx_train': np.array(data_idx_train),
                                         'data_idx_valid': np.array(data_idx_valid),
                                         'data_idx_test': np.array(data_idx_test)}})
        self.splits_info.update({split_name: {'n_train': n_train,
                                              'n_valid': n_valid,
                                              'n_test': n_test,
                                              'n_data': self.n_data,
                                              'r_cut': r_cut,
                                              'mic': mic,
                                              'precomputed_graph': False}})
        self.data_split = d

    def _record_split(self,
                      data_idx_train,
                      data_idx_valid,
                      data_idx_test,
                      split_name,
                      r_cut,
                      mic,
                      precomputed_graph):
        # Store exact indices so deterministic precomputed graph splits remain reproducible.
        self.splits.update({split_name: {'data_idx_train': np.array(data_idx_train),
                                         'data_idx_valid': np.array(data_idx_valid),
                                         'data_idx_test': np.array(data_idx_test)}})

        # Retain the model cutoff even though no runtime neighbor construction occurs.
        self.splits_info.update({split_name: {'n_train': len(data_idx_train),
                                              'n_valid': len(data_idx_valid),
                                              'n_test': len(data_idx_test),
                                              'n_data': self.n_data,
                                              'r_cut': r_cut,
                                              'mic': mic,
                                              'precomputed_graph': precomputed_graph}})

    def _validate_precomputed_graph(self, r_cut):
        # Check explicit sidecar values when the caller supplied adjacent dataset metadata.
        if self.graph_metadata is not None:
            validate_precomputed_graph_metadata(metadata=self.graph_metadata, r_cut=r_cut)

        # Resolve semantic properties so custom NPZ key mappings remain supported.
        required_properties = (pn.idx_i, pn.idx_j, pn.pair_mask, pn.bond_prob, pn.bond_mask)
        missing_properties = [name for name in required_properties if self.prop_keys.get(name) not in self.data]
        if missing_properties:
            raise ValueError(f'Precomputed graphs require properties {missing_properties}.')

        idx_i = np.asarray(self.data[self.prop_keys[pn.idx_i]])
        idx_j = np.asarray(self.data[self.prop_keys[pn.idx_j]])
        pair_mask = np.asarray(self.data[self.prop_keys[pn.pair_mask]])
        bond_prob = np.asarray(self.data[self.prop_keys[pn.bond_prob]])
        bond_mask = np.asarray(self.data[self.prop_keys[pn.bond_mask]])
        z = np.asarray(self.data[self.prop_keys[pn.atomic_type]])

        # All edge arrays must share the same batch and padded-edge dimensions.
        if idx_i.ndim != 2 or idx_j.shape != idx_i.shape:
            raise ValueError('`idx_i` and `idx_j` must have aligned shape (B, P).')
        if pair_mask.shape != idx_i.shape or bond_mask.shape != idx_i.shape:
            raise ValueError('`pair_mask` and `bond_mask` must have aligned shape (B, P).')
        if bond_prob.shape != (*idx_i.shape, 4):
            raise ValueError('`bond_prob` must have aligned shape (B, P, 4).')
        if idx_i.shape[0] != self.n_data or z.shape[0] != self.n_data:
            raise ValueError('Precomputed graph arrays must share the dataset batch dimension.')

        # Masks are boolean contracts even when stored as integer or floating NPZ arrays.
        if not np.isin(pair_mask, (0, 1)).all() or not np.isin(bond_mask, (0, 1)).all():
            raise ValueError('`pair_mask` and `bond_mask` must contain only 0/1 values.')
        pair_valid = pair_mask.astype(bool)
        bond_valid = bond_mask.astype(bool)
        if np.logical_and(bond_valid, ~pair_valid).any():
            raise ValueError('`bond_mask` must be a subset of `pair_mask`.')

        # Directed indices must be integers in bounds, while every padded edge uses the -1 sentinel.
        if not np.issubdtype(idx_i.dtype, np.integer) or not np.issubdtype(idx_j.dtype, np.integer):
            raise ValueError('Precomputed `idx_i` and `idx_j` arrays must use an integer dtype.')
        if (idx_i[~pair_valid] != -1).any() or (idx_j[~pair_valid] != -1).any():
            raise ValueError('Every edge with `pair_mask == 0` must use -1 for both directed indices.')
        n_atoms = z.shape[1]
        if ((idx_i[pair_valid] < 0) | (idx_i[pair_valid] >= n_atoms)).any():
            raise ValueError('Valid `idx_i` entries must be directed atom indices within [0, N).')
        if ((idx_j[pair_valid] < 0) | (idx_j[pair_valid] >= n_atoms)).any():
            raise ValueError('Valid `idx_j` entries must be directed atom indices within [0, N).')

        # Valid edges may not reference zero-padded atoms.
        safe_i = np.where(pair_valid, idx_i, 0)
        safe_j = np.where(pair_valid, idx_j, 0)
        if (np.take_along_axis(z, safe_i, axis=1)[pair_valid] == 0).any():
            raise ValueError('Valid `idx_i` entries may not reference padded atoms.')
        if (np.take_along_axis(z, safe_j, axis=1)[pair_valid] == 0).any():
            raise ValueError('Valid `idx_j` entries may not reference padded atoms.')

        # Bond probabilities must be finite, nonnegative, and normalized on annotated bonded edges.
        if not np.isfinite(bond_prob).all() or (bond_prob < 0).any():
            raise ValueError('`bond_prob` values must be finite and nonnegative.')
        probability_tolerance = 1e-5
        if self.graph_metadata is not None:
            probability_tolerance = float(self.graph_metadata.get('bond_probability_tolerance',
                                                                  probability_tolerance))
        bonded_sums = bond_prob.sum(axis=-1)[bond_valid]
        if not np.isclose(bonded_sums, 1.0, atol=probability_tolerance, rtol=0).all():
            raise ValueError('Bonded `bond_prob` rows must sum to one within the metadata tolerance.')

        # Validate every supplied raw state descriptor against the same fixed edge graph. The model constructs
        # relative 12-channel descriptors later; the persisted NPZ contract remains four probabilities per state.
        state_descriptors = {}
        for state in range(3):
            prob_name = getattr(pn, f'bond_prob_s{state}')
            mask_name = getattr(pn, f'bond_mask_s{state}')
            prob_key = self.prop_keys.get(prob_name)
            mask_key = self.prop_keys.get(mask_name)
            prob_present = prob_key in self.data
            mask_present = mask_key in self.data
            if prob_present != mask_present:
                raise ValueError(f'State {state} requires both `{prob_key}` and `{mask_key}`.')
            if not prob_present:
                continue

            state_prob = np.asarray(self.data[prob_key])
            state_mask = np.asarray(self.data[mask_key])
            if state_prob.shape != (*idx_i.shape, 4):
                raise ValueError(f'`{prob_key}` must have aligned shape (B, P, 4).')
            if state_mask.shape != idx_i.shape:
                raise ValueError(f'`{mask_key}` must have aligned shape (B, P).')
            if not np.isin(state_mask, (0, 1)).all():
                raise ValueError(f'`{mask_key}` must contain only 0/1 values.')

            state_valid = state_mask.astype(bool)
            if np.logical_and(state_valid, ~pair_valid).any():
                raise ValueError(f'`{mask_key}` must be a subset of `pair_mask`.')
            if not np.isfinite(state_prob).all() or (state_prob < 0).any():
                raise ValueError(f'`{prob_key}` values must be finite and nonnegative.')
            state_sums = state_prob.sum(axis=-1)[state_valid]
            if not np.isclose(state_sums, 1.0, atol=probability_tolerance, rtol=0).all():
                raise ValueError(f'Annotated `{prob_key}` rows must sum to one within the metadata tolerance.')
            state_descriptors[state] = (state_prob, state_mask)

        if 0 in state_descriptors:
            state_0_prob, state_0_mask = state_descriptors[0]
            if not np.array_equal(bond_prob, state_0_prob) or not np.array_equal(bond_mask, state_0_mask):
                raise ValueError('Canonical `bond_prob`/`bond_mask` must be exact aliases of state-0 descriptors.')

    def random_split(self,
                     n_train,
                     n_valid,
                     n_test,
                     r_cut=None,
                     mic=None,
                     seed=0,
                     training=True,
                     precomputed_graph: bool = False):

        idx_train, idx_valid, idx_test = self.generate_split_indices(self.data,
                                                                     n_train=n_train,
                                                                     n_valid=n_valid,
                                                                     n_test=n_test,
                                                                     seed=seed
                                                                     )

        self.index_split(data_idx_train=idx_train,
                         data_idx_valid=idx_valid,
                         data_idx_test=idx_test,
                         training=training,
                         r_cut=r_cut,
                         mic=mic,
                         precomputed_graph=precomputed_graph)

    @staticmethod
    def generate_split_indices(data, n_train, n_valid, n_test=None, seed=0, draw_strat=None):
        set_seeds(seed)
        _k = list(data.keys())[0]
        n_data = len(data[_k])
        perm = np.random.RandomState(seed).permutation(n_data)
        idx_all = np.arange(n_data)[perm]

        if draw_strat:
            raise NotImplementedError('Stratified data set sampling is not implemented.')
            # idx_train = draw_strat_sample(data[draw_strat][perm],
            #                               n=n_train)
            # idx_valid = np.array(list(set(idx_all) - set(idx_train)))
            # # set sorts the indices, so we have to permute them again
            # valid_perm = np.random.RandomState(seed).permutation(len(idx_valid))
            # idx_valid = idx_valid[valid_perm][:n_valid]
        else:
            idx = idx_all[:n_train + n_valid]
            idx_valid, idx_train = np.split(idx, indices_or_sections=[n_valid])

        # set sorts the indices, so we have to permute them again
        idx_test = np.array(list(set(idx_all) - set(idx_train) - set(idx_valid)))
        test_perm = np.random.RandomState(seed).permutation(len(idx_test))
        idx_test = idx_test[test_perm][:n_test]  # array[:None] returns all elements of array

        if n_test is None:
            n_test = len(idx_test)

        # assert no duplicates per subset
        assert len(set(idx_train)) == n_train
        assert len(set(idx_valid)) == n_valid
        assert len(set(idx_test)) == n_test
        # make sure there is no overlap
        assert len(set(idx_test) & set(idx_train)) == 0
        assert len(set(idx_test) & set(idx_valid)) == 0
        assert len(set(idx_train) & set(idx_valid)) == 0

        return idx_train, idx_valid, idx_test

    def shift_x_by_mean_x(self, x):
        n_atoms = self.data[self.prop_keys[pn.atomic_type]].shape[-1]
        if x in self.track_shift_x_by_mean_x:
            logging.warning(f'You already called `shift_x_by_mean` for `x={x}`. It is not shifted again.')
        else:
            p_key = self.prop_keys[x]
            p_mean = self.data_split['train'][p_key].reshape(-1).mean()
            self.data_split['train'][p_key] -= p_mean
            self.data_split['valid'][p_key] -= p_mean
            self.data_split['test'][p_key] -= p_mean
            self.scales[x]['per_atom_shift'] = [0] + [p_mean / n_atoms] * 100
            self.track_shift_x_by_mean_x += [x]

    def divide_x_by_std_y(self, x, y):
        if x in self.track_divide_x_by_std_y:
            logging.warning(f'You already called `divide_x_by_std_y` for `x={x}`. It is not divided again.')
        else:
            x_key = self.prop_keys[x]
            y_key = self.prop_keys[y]
            y_scale = self.data_split['train'][y_key].reshape(-1).std()
            self.data_split['train'][x_key] /= y_scale
            self.data_split['valid'][x_key] /= y_scale
            self.data_split['test'][x_key] /= y_scale
            self.scales[x]['scale'] = y_scale.item()
            self.track_divide_x_by_std_y += [x]

    def shift_x_by_type(self, x, shifts=None):
        if shifts is not None:
            self.shift_x_by_type_hand(x, shifts=shifts)
        else:
            self.shift_x_by_type_lse(x)

    def shift_x_by_type_hand(self, x, shifts: Dict[int, float]):
        if x in self.track_shift_x_by_type:
            logging.warning(f'You already called `shift_x_by_type` for `x={x}`. It is not shifted again.')
        else:
            x_key = self.prop_keys[x]
            z_key = self.prop_keys[pn.atomic_type]

            shifts_arr = np.zeros(int(max(list(shifts.keys()))) + 1)
            for k, v in shifts.items():
                shifts_arr[k] = v

            def apply_shifts(q, _z):
                q_scaled = q - np.take(shifts_arr, _z).sum(axis=-1)
                return q_scaled

            self.data_split['train'][x_key] = apply_shifts(self.data_split['train'][x_key].reshape(-1),
                                                           self.data_split['train'][z_key]).reshape(
                self.data_split['train'][x_key].shape)

            self.data_split['valid'][x_key] = apply_shifts(self.data_split['valid'][x_key].reshape(-1),
                                                           self.data_split['valid'][z_key]).reshape(
                self.data_split['valid'][x_key].shape)

            self.scales[x]['per_atom_shift'] = shifts_arr.reshape(-1).tolist()
            self.track_shift_x_by_type += [x]

    def shift_x_by_type_lse(self, x):
        if x in self.track_shift_x_by_type:
            logging.warning(f'You already called `shift_x_by_type` for `x={x}`. It is not shifted again.')
        else:
            x_key = self.prop_keys[x]
            z_key = self.prop_keys[pn.atomic_type]

            z = self.data_split['train'][z_key]

            shifts, x_shift_lse = get_per_atom_shift(z=z,
                                                     q=self.data_split['train'][x_key].reshape(-1),
                                                     pad_value=0)

            def apply_shifts(q, _z):
                q_scaled = q - np.take(shifts, _z).sum(axis=-1)
                return q_scaled

            self.data_split['train'][x_key] = x_shift_lse.reshape(self.data_split['train'][x_key].shape)
            self.data_split['valid'][x_key] = apply_shifts(self.data_split['valid'][x_key].reshape(-1),
                                                           self.data_split['valid'][z_key]).reshape(
                self.data_split['valid'][x_key].shape)

            self.scales[x]['per_atom_shift'] = shifts.reshape(-1).tolist()
            self.track_shift_x_by_type += [x]

    def get_data_split(self):
        return self.data_split

    def load_split(self,
                   file,
                   r_cut,
                   split_name,
                   mic=None,
                   n_train=None,
                   n_valid=None,
                   n_test=None,
                   precomputed_graph: bool = False):
        path, filename = os.path.split(file)
        split_idx = self.load_splits_from_file(path=path, filename=filename)[split_name]
        key_2_n = {'data_idx_train': n_train, 'data_idx_valid': n_valid, 'data_idx_test': n_test}

        valid_keys = list(key_2_n.keys())

        subset_split = {k: v[:key_2_n[k]] for (k, v) in split_idx.items() if k in valid_keys}
        self.index_split(r_cut=r_cut,
                         mic=mic,
                         training=False,
                         precomputed_graph=precomputed_graph,
                         **subset_split)

    def save_splits_to_file(self, path, filename):
        splits_ = tree_map(lambda y: y.tolist(), self.splits)
        save_dict(path=path, filename=filename, data=splits_, exists_ok=True)
        print('Saved the data indices of the splits to {}'.format(os.path.join(path, filename)))

    @staticmethod
    def load_splits_from_file(path, filename):
        _data_idx = read_json(path=os.path.join(path, filename))
        return _data_idx

    def __dict_repr__(self):
        return {'dataset': self.splits_info}


def tree_map_by_key(fn, x, keys):
    x_flat = flatten_dict(x)
    apply_mask = unflatten_dict({p: (p[-1] in keys) for p in x_flat})
    msk_fn = lambda y, m: fn(y) if m else y
    return tree_map(msk_fn, x, apply_mask)


def node_mask_required(z):
    """
    Check if node mask needs to passed explicitly.

    Args:
        z (Array): Array with the atomic types, shape: (B,n)

    Returns:

    """

    if z.min() < 1:
        return True
    if z.max() == np.inf:
        return True

    return False
