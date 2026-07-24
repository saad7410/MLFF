import numpy as np
import jax
import jax.numpy as jnp
import logging
import os
import argparse
import wandb
import json
import portpicker

try:
    from jax import tree as _jtree
    tree_map = _jtree.map          # JAX ≥ 0.4.25 (incl. 0.6.x)
except Exception:
    from jax.tree_util import tree_map

from pathlib import Path
from typing import Dict
from ase.units import *

from mlff.io import (checkpoint_fingerprint, create_directory, bundle_dicts,
                     save_dict, read_json, load_checkpoint_identity,
                     load_params_from_ckpt_dir)
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.data import DataTuple, DataSet, load_precomputed_graph_metadata, select_data_for_model
from mlff.cAPI.process_argparse import StoreDictKeyPair
from mlff.nn.stacknet import (get_obs_and_force_fn, get_observable_fn,
                              get_energy_force_stress_fn, get_delta_energy_force_fn,
                              get_delta_offset_energy_force_fn,
                              init_stack_net)
from mlff.nn import (So3krates, build_delta_offset_targets,
                     init_state_specific_delta_so3krates,
                     init_state_routed_offset_so3krates,
                     get_pretrained_backbone_paths, init_delta_model,
                     init_delta_offset_model, load_pretrained_backbone,
                     restore_ground_prediction_units,
                     upgrade_stacknet_for_bond_delta,
                     upgrade_stacknet_for_relative_bond_delta)
from mlff.nn.representation.delta import (ABSOLUTE_BOND_DESCRIPTOR,
                                          ABSOLUTE_BOND_FEATURE_DIM,
                                          DELTA_OFFSET_TRAINING_MODE,
                                          LEGACY_BOND_LAYOUT,
                                          NAMED_BOND_LAYOUT,
                                          OFFSET_SIGN_CONVENTION,
                                          RELATIVE_BOND_DESCRIPTOR,
                                          RELATIVE_BOND_FEATURE_DIM)
from mlff.nn.observable import Energy
from mlff.data import AseDataLoader
from mlff.properties import delta_offset_property_keys, md17_property_keys

import mlff.properties.property_names as pn

DELTA_OFFSET_GROUP_KEY = '_delta_offset_geometry_group'


def unit_convert_data(x: Dict, table: Dict):
    """
    Convert units in the data dictionary.

    Args:
        x (Dict): The data dictionary.
        table (Dict): A dictionary mapping quantities to conversion factors.

    Returns: The data dictionary with converted quantities.

    """
    for (k, v) in x.items():
        if k in list(table.keys()):
            logging.info('Converted {} to ase default unit.'.format(k))
            x[k] *= table[k]
    return x


def is_bond_aware_stacknet_metadata(h: Dict) -> bool:
    # Infer representation compatibility from serialized layers so delta mode cannot drift from its checkpoint.
    return any(layer_h.get('so3krates_layer', {}).get('bond_aware', False)
               for layer_h in h['stack_net']['layers'])


def validate_and_filter_delta_offset_data(data: Dict, prop_keys: Dict):
    """Validate zero-based active labels and retain only S1/S2 rows."""
    required_properties = (pn.atomic_position, pn.atomic_type, pn.energy,
                           pn.force, pn.active_state)
    missing_mappings = [name for name in required_properties if name not in prop_keys]
    if missing_mappings:
        raise KeyError(f'Missing delta-offset property mappings {missing_mappings}.')
    missing_arrays = [prop_keys[name] for name in required_properties
                      if prop_keys[name] not in data]
    if missing_arrays:
        raise KeyError(f'Delta-offset data is missing required arrays {missing_arrays}.')

    state_key = prop_keys[pn.active_state]
    states = np.asarray(data[state_key])
    if states.ndim == 2 and states.shape[1] == 1:
        states = states[:, 0]
    elif states.ndim != 1:
        raise ValueError(f'`{state_key}` must have shape (B,) or (B, 1).')
    if not np.issubdtype(states.dtype, np.integer):
        raise TypeError(f'`{state_key}` must contain integer zero-based state IDs.')
    if not np.isin(states, (0, 1, 2)).all():
        invalid = np.unique(states[~np.isin(states, (0, 1, 2))]).tolist()
        raise ValueError(f'`{state_key}` contains unsupported state IDs {invalid}; expected 0, 1, or 2.')

    n_rows = len(states)
    R_key = prop_keys[pn.atomic_position]
    z_key = prop_keys[pn.atomic_type]
    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    for key in (R_key, energy_key, force_key):
        if np.asarray(data[key]).shape[0] != n_rows:
            raise ValueError(f'`{key}` must have the same leading dimension as `{state_key}`.')

    # A fixed-composition file may store z once as (N,). Normalize it here so
    # filtering and later train/valid concatenation remain row aligned.
    normalized = {key: np.asarray(value) for key, value in data.items()}
    if normalized[z_key].ndim == 1:
        normalized[z_key] = np.repeat(normalized[z_key][None, :], n_rows, axis=0)
    elif normalized[z_key].shape[0] != n_rows:
        raise ValueError(f'`{z_key}` must have shape (N,) or a leading data dimension of {n_rows}.')
    normalized[state_key] = states[:, None]

    excited_mask = states != 0
    if not excited_mask.any():
        raise ValueError('Delta-offset training data contains no excited-state rows.')
    filtered = {}
    for key, value in normalized.items():
        if value.ndim > 0 and value.shape[0] == n_rows:
            filtered[key] = value[excited_mask]
        else:
            filtered[key] = value

    selected_energy = np.asarray(filtered[energy_key])
    selected_force = np.asarray(filtered[force_key])
    n_excited = int(excited_mask.sum())
    if selected_energy.shape not in ((n_excited,), (n_excited, 1)):
        raise ValueError('Active energies must have shape (B,) or (B, 1).')
    if selected_force.shape != np.asarray(filtered[R_key]).shape:
        raise ValueError('Active forces must have shape (B, N, 3) matching positions.')
    if not np.isfinite(selected_energy).all():
        raise ValueError('Active energies must be finite on every selected excited-state row.')
    node_mask_key = prop_keys[pn.node_mask]
    if node_mask_key in filtered:
        real_atoms = np.asarray(filtered[node_mask_key]).astype(bool)
    else:
        real_atoms = np.asarray(filtered[z_key]) != 0
    if selected_force.shape[:2] != real_atoms.shape:
        raise ValueError('Active force arrays must align with the structure/atom dimensions.')
    invalid_real_force = (~np.isfinite(selected_force)
                          & np.broadcast_to(real_atoms[..., None], selected_force.shape))
    if invalid_real_force.any():
        raise ValueError('Active forces must be finite on every non-padded atom.')
    state_ids = tuple(int(state) for state in np.unique(states[excited_mask]))
    return filtered, state_ids


def inspect_delta_offset_files(data_files, prop_keys):
    """Inspect only numeric arrays needed to choose the routed offset contract."""
    state_ids = set()
    relative_available = True
    absolute_available = True
    state_key = prop_keys[pn.active_state]
    core_keys = tuple(prop_keys[name] for name in
                      (pn.atomic_position, pn.atomic_type, pn.energy, pn.force, pn.active_state))

    for path in data_files:
        if Path(path).suffix != '.npz':
            raise ValueError('`--delta_offset` accepts NPZ data with R/z/E/F/astate labels.')
        with np.load(path, allow_pickle=False) as archive:
            available = set(archive.files)
            missing = [key for key in core_keys if key not in available]
            if missing:
                raise KeyError(f'Delta-offset file {path} is missing required arrays {missing}.')
            states = np.asarray(archive[state_key])
            if states.ndim == 2 and states.shape[1] == 1:
                states = states[:, 0]
            if states.ndim != 1 or not np.issubdtype(states.dtype, np.integer):
                raise TypeError(f'`{state_key}` in {path} must be an integer (B,) or (B, 1) array.')
            if not np.isin(states, (0, 1, 2)).all():
                invalid = np.unique(states[~np.isin(states, (0, 1, 2))]).tolist()
                raise ValueError(f'`{state_key}` in {path} contains unsupported state IDs {invalid}.')
            file_state_ids = set(int(state) for state in np.unique(states) if state != 0)
            if not file_state_ids:
                raise ValueError(f'Delta-offset file {path} contains no S1/S2 rows.')
            state_ids.update(file_state_ids)

            absolute_keys = (prop_keys[pn.bond_prob], prop_keys[pn.bond_mask])
            absolute_available &= all(key in available for key in absolute_keys)
            relative_keys = [prop_keys[pn.bond_prob_s0], prop_keys[pn.bond_mask_s0]]
            for state in file_state_ids:
                relative_keys.extend((prop_keys[getattr(pn, f'bond_prob_s{state}')],
                                      prop_keys[getattr(pn, f'bond_mask_s{state}')]))
            relative_available &= all(key in available for key in relative_keys)
    return tuple(sorted(state_ids)), relative_available, absolute_available


def _geometry_group_ids(archive, prop_keys):
    """Build stable row groups so repeated electronic-state geometries cannot leak."""
    R = np.asarray(archive[prop_keys[pn.atomic_position]])
    n_rows = len(R)
    z = np.asarray(archive[prop_keys[pn.atomic_type]])
    if z.ndim == 1:
        z = np.repeat(z[None, :], n_rows, axis=0)
    if z.shape[0] != n_rows:
        raise ValueError('Atomic types cannot be aligned to geometry rows for grouped splitting.')
    # Group only genuinely identical geometries. Source frame identifiers are
    # not reliable geometry identities outside I02 (e.g. A01 can reuse them
    # for rows with different coordinates).
    raw_keys = [np.ascontiguousarray(R[row]).tobytes()
                + np.ascontiguousarray(z[row]).tobytes()
                for row in range(n_rows)]

    key_to_group = {}
    groups = np.empty(n_rows, dtype=np.int64)
    for row, key in enumerate(raw_keys):
        groups[row] = key_to_group.setdefault(key, len(key_to_group))
    return groups[:, None]


def generate_grouped_split_indices(group_ids, n_train, n_valid, n_test=None, seed=0):
    """Split whole geometry groups while honoring requested row counts exactly."""
    groups = np.asarray(group_ids).reshape(-1)
    n_rows = len(groups)
    if n_train is None or n_valid is None:
        raise ValueError('Grouped splitting requires explicit n_train and n_valid.')
    if n_train < 0 or n_valid < 0 or n_train + n_valid > n_rows:
        raise ValueError(f'Invalid grouped split sizes for {n_rows} excited-state rows.')

    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    unique_groups = unique_groups[rng.permutation(len(unique_groups))]
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}

    def take_exact(candidates, target, split_name, prefer_larger=False):
        if prefer_larger:
            candidates = sorted(
                candidates,
                key=lambda group: len(group_rows[group]),
                reverse=True)
        selected = []
        remaining = []
        count = 0
        for group in candidates:
            size = len(group_rows[group])
            if count + size <= target:
                selected.append(group)
                count += size
            else:
                remaining.append(group)
        if count != target:
            sizes = sorted({len(group_rows[group]) for group in candidates})
            raise ValueError(f'`{split_name}` size {target} cannot be formed without splitting '
                             f'geometry groups with sizes {sizes}.')
        return selected, remaining

    def allocate(prefer_larger):
        train_groups, remaining = take_exact(
            list(unique_groups), n_train, 'n_train', prefer_larger=prefer_larger)
        valid_groups, remaining = take_exact(
            remaining, n_valid, 'n_valid', prefer_larger=prefer_larger)
        remaining_count = sum(len(group_rows[group]) for group in remaining)
        resolved_n_test = remaining_count if n_test is None else n_test
        if resolved_n_test < 0 or resolved_n_test > remaining_count:
            raise ValueError(
                f'Invalid n_test={resolved_n_test} for {remaining_count} remaining grouped rows.')
        test_groups, _ = take_exact(
            remaining, resolved_n_test, 'n_test', prefer_larger=prefer_larger)
        return train_groups, valid_groups, test_groups

    try:
        # Keep the shuffled group order when exact allocation is straightforward.
        # This distributes repeated geometries across splits instead of placing
        # every larger group in training.
        train_groups, valid_groups, test_groups = allocate(prefer_larger=False)
    except ValueError:
        # Some group-size combinations need the established larger-first
        # fallback to reach the requested row counts exactly.
        train_groups, valid_groups, test_groups = allocate(prefer_larger=True)

    def rows_for(selected):
        if not selected:
            return np.asarray([], dtype=np.int64)
        rows = np.concatenate([group_rows[group] for group in selected])
        return rows[rng.permutation(len(rows))]

    return rows_for(train_groups), rows_for(valid_groups), rows_for(test_groups)


def load_delta_offset_npz(path, inputs, prop_keys, conversion_table,
                          bond_descriptor_mode=ABSOLUTE_BOND_DESCRIPTOR):
    """Load only required numeric arrays, leaving object provenance untouched."""
    properties = tuple(dict.fromkeys((*inputs, pn.energy, pn.force, pn.active_state)))
    generated = {pn.idx_i, pn.idx_j, pn.node_mask, pn.cell_offset}
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        aliases = {}
        if bond_descriptor_mode == RELATIVE_BOND_DESCRIPTOR:
            # The canonical arrays seen by fixed-graph validation represent b0
            # in relative mode, even if the source also stores per-row active
            # descriptors under bond_prob/bond_mask.
            aliases[prop_keys[pn.bond_prob]] = prop_keys[pn.bond_prob_s0]
            aliases[prop_keys[pn.bond_mask]] = prop_keys[pn.bond_mask_s0]
        missing = []
        for name in properties:
            key = prop_keys[name]
            source_key = aliases.get(key, key)
            if name not in generated and source_key not in available:
                missing.append(key)
        if missing:
            raise KeyError(f'Delta-offset file {path} is missing model arrays {missing}.')
        data = {}
        for name in properties:
            key = prop_keys[name]
            source_key = aliases.get(key, key)
            if source_key in available:
                data[key] = np.asarray(archive[source_key])
        data[DELTA_OFFSET_GROUP_KEY] = _geometry_group_ids(archive, prop_keys)
    data = unit_convert_data(data, table=conversion_table)
    return validate_and_filter_delta_offset_data(data=data, prop_keys=prop_keys)


def generate_delta_offset_targets(data_split,
                                  teacher_net,
                                  teacher_params,
                                  teacher_scales,
                                  prop_keys,
                                  batch_size):
    """Populate raw-unit Offset_E/F targets using a fixed, external teacher."""
    if batch_size <= 0:
        raise ValueError('Teacher prediction batch size must be positive.')
    teacher_fn = jax.jit(jax.vmap(get_obs_and_force_fn(teacher_net), in_axes=(None, 0)))
    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    offset_energy_key = prop_keys[pn.offset_energy]
    offset_force_key = prop_keys[pn.offset_force]
    pred_energy_key = prop_keys[pn.pred_energy_0]
    pred_force_key = prop_keys[pn.pred_force_0]

    for split_name, split in data_split.items():
        n_rows = len(split[energy_key])
        pred_energy_batches = []
        pred_force_batches = []
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            batch_inputs = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value[start:stop]), split)
            teacher_outputs = teacher_fn(teacher_params, batch_inputs)
            pred_energy, pred_force = restore_ground_prediction_units(
                outputs=teacher_outputs,
                inputs=batch_inputs,
                scales=teacher_scales,
                prop_keys=prop_keys)
            pred_energy_batches.append(np.asarray(pred_energy))
            pred_force_batches.append(np.asarray(pred_force))

        if n_rows:
            pred_energy = np.concatenate(pred_energy_batches, axis=0)
            pred_force = np.concatenate(pred_force_batches, axis=0)
        else:
            pred_energy = np.empty_like(split[energy_key])
            pred_force = np.empty_like(split[force_key])
        offset_energy, offset_force = build_delta_offset_targets(
            active_energy=split[energy_key],
            active_force=split[force_key],
            pred_energy_0=pred_energy,
            pred_force_0=pred_force)
        split[offset_energy_key] = np.asarray(offset_energy)
        split[offset_force_key] = np.asarray(offset_force)
        # Retain the exact generated baseline for audit/debugging, but these
        # arrays are not model inputs and are never part of the optimizer tree.
        split[pred_energy_key] = pred_energy
        split[pred_force_key] = pred_force
    return data_split


def train_so3krates():
    # Avoid distributed-runtime side effects when importing data/model helpers
    # (notably in evaluators and tests); CLI execution retains the same setup.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

    # Create the parser
    parser = argparse.ArgumentParser(description='Train a So3krates model.')

    parser.add_argument("--prop_keys", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        default=md17_property_keys,
                        help='Property keys of the data set. Needs only to be specified, if e.g. the keys of the '
                             'properties in the data set that the model is applied to differ from the keys the model'
                             'has been trained on.')

    # Add the arguments
    parser.add_argument('--data_file', type=str, required=False, default=None)
    parser.add_argument('--train_data_file', type=str, required=False, default=None)
    parser.add_argument('--valid_data_file', type=str, required=False, default=None)

    parser.add_argument('--shift_by', type=str, required=False, default='mean',
                        metavar='Possible values: mean, atomic_number, lse')

    parser.add_argument('--shifts', action=StoreDictKeyPair, required=False, default=None,
                        metavar="1=-100.5,6=-550.2,...")

    parser.add_argument('--ckpt_dir', type=str, required=False, default=None,
                        help='Path to the checkpoint directory (must not exist). '
                             'If not set, defaults to `current_directory/module`.')

    parser.add_argument('--ckpt_manager_options', type=json.loads, required=False, default=None,
                        metavar='{"key": value, "key1": value1, ...}',
                        help='Options for the checkpoint manager. See '
                             'https://github.com/google/orbax/blob/main/docs/checkpoint.md for all options.')

    parser.add_argument('--restart_from_ckpt_dir', type=str, required=False, default=None,
                        help='Path to a checkpoint directory from which to load model parameters and start the '
                             'training.')

    transfer_mode = parser.add_mutually_exclusive_group()
    transfer_mode.add_argument(
        '--delta', action='store_true', required=False,
        help='Train state-1/state-2 physical corrections from paired ground-state labels.')
    transfer_mode.add_argument(
        '--delta_offset', action='store_true', required=False,
        help='Train routed S1/S2 offsets from active labels and a frozen ground-model baseline.')
    parser.add_argument('--pretrained_ground_ckpt_dir', type=str, required=False, default=None,
                        help='Ground-state SO3krates checkpoint used to initialize the shared delta backbone.')
    parser.add_argument('--freeze_pretrained_backbone', action='store_true', required=False,
                        help='Freeze only transferred SO3krates parameters; newly added bond branches and the delta '
                             'head remain trainable.')

    # Model Arguments
    parser.add_argument('--r_cut', type=float, required=False, default=5., help='Local neighborhood cutoff.')
    parser.add_argument('--F', type=int, required=False, default=132, help='Feature dimension.')
    parser.add_argument('--L', type=int, required=False, default=3, help='Number of layers.')
    parser.add_argument('--H', type=int, required=False, default=4, help='Number of heads.')
    parser.add_argument('--degrees', nargs='+', type=int, required=False, default=[1, 2, 3],
                        help='Degrees for the spherical harmonic coordinates.')

    parser.add_argument('--bond_aware', action='store_true', required=False,
                        help='Enable invariant bond conditioning from a precomputed NPZ graph.')

    parser.add_argument('--so3krates_layer_kwargs', type=json.loads, required=False, default=None,
                        metavar='{"key": value, "key1": value1, ...}',
                        help='Additional options for SO3krates layer.'
                        )

    parser.add_argument('--zbl_repulsion', action="store_true", required=False,
                        help='Add ZBL repulsion to learned PES.')
    parser.add_argument('--geometry_embed_kwargs', type=json.loads, required=False, default=None,
                        metavar='{"key": value, "key1": value1, ...}',
                        help='Keyword arguments that should be passed to `GeometryEmbed` module.')

    # Structure arguments
    parser.add_argument('--mic', action="store_true", required=False,
                        help='If minimal image convention should be applied.')

    # Data Arguments
    parser.add_argument('--n_train', type=int, required=False, help='Number of training points.', default=None)
    parser.add_argument('--n_valid', type=int, required=False, help='Number of validation points.', default=None)
    parser.add_argument('--n_test', type=int, required=False, help='Number of test points.', default=None)

    parser.add_argument('--epochs', type=int, required=False, help='Number of training epochs.')
    parser.add_argument('--steps', type=int, required=False, help='Number of training steps.')
    parser.add_argument('--lr_stop', type=float, required=False, default=1e-5,
                        help='Stop training if learning rate is smaller than given learning rate.')

    # Arguments that determine the training parameters
    parser.add_argument('--batch_size', type=int, required=False, default=None,
                        help="Batch size for training and validation.")
    parser.add_argument('--training_batch_size', type=int, required=False, default=None,
                        help="Batch size for training (gradient calculation). Defaults to batch_size if not filled.")
    parser.add_argument('--validation_batch_size', type=int, required=False, default=None,
                        help="Batch size of the validation pass. Defaults to batch_size if not filled.")
    parser.add_argument("--units", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default=None,
                        help='Units in the data set for the quantities. Needs only to be specified'
                             'if the model has been trained on units different from the ones present in the data set.')

    # Training arguments
    parser.add_argument('--model_seed', type=int, required=False, default=0)
    parser.add_argument('--data_seed', type=int, required=False, default=0)
    parser.add_argument('--training_seed', type=int, required=False, default=0)

    parser.add_argument('--targets', nargs='+', required=False, default=[pn.energy, pn.force])
    parser.add_argument('--inputs', nargs='+', required=False, default=[pn.atomic_type,
                                                                        pn.atomic_position,
                                                                        pn.idx_i, pn.idx_j,
                                                                        pn.node_mask])

    parser.add_argument('--lr', type=float, required=False, default=1e-3)
    parser.add_argument('--lr_decay_plateau', action=StoreDictKeyPair, required=False, default=None)

    lr_decay_exp_default = {'transition_steps': 100_000, 'decay_factor': 0.7}
    parser.add_argument('--lr_decay_exp', action=StoreDictKeyPair, required=False, default=lr_decay_exp_default)
    parser.add_argument('--lr_warmup', action=StoreDictKeyPair, required=False, default=None)

    parser.add_argument('--clip_by_global_norm', type=float, required=False, default=None)

    # Split the established energy/force weighting evenly across the two delta states.
    default_loss_weights = {pn.energy: 0.01,
                            pn.force: 0.99,
                            pn.stress: 0.01,
                            pn.delta_energy_1: 0.005,
                            pn.delta_energy_2: 0.005,
                            pn.delta_force_1: 0.495,
                            pn.delta_force_2: 0.495,
                            pn.offset_energy: 0.01,
                            pn.offset_force: 0.99}
    parser.add_argument('--loss_weights', action=StoreDictKeyPair, required=False, default=default_loss_weights)
    parser.add_argument("--loss_variance_scaling", action="store_true",
                        help="Scale the individual loss terms by the inverse of their variance in the training split. "
                             "Loss weights specified via the --loss_weights keyword are still used.")

    parser.add_argument('--eval_every_t', type=int, required=False, default=None,
                        help='Evaluate the model every t steps. Defaults to the number of steps that correspond to '
                             'evaluation after every epoch.')
    parser.add_argument('--use_wandb', type=bool, required=False, default=True)

    parser.add_argument('--wandb_init', action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default={})

    parser.add_argument('--jax_dtype', type=str, required=False, default='x32',
                        help='Set JAX default dtype. Default is jax.numpy.float32')

    args = parser.parse_args()

    # Precision must be configured before checkpoint arrays are restored or
    # hashed, otherwise an x64 teacher can be truncated before identity checks.
    jax_dtype = args.jax_dtype
    if jax_dtype == 'x64':
        from jax import config
        config.update("jax_enable_x64", True)

    # Copy parser defaults before adding opt-in delta and bond properties to avoid mutating the shared mapping.
    prop_keys = dict(args.prop_keys)
    for property_name, default_key in md17_property_keys.items():
        prop_keys.setdefault(property_name, default_key)

    delta_learning = args.delta
    delta_offset_learning = args.delta_offset
    transfer_learning = delta_learning or delta_offset_learning
    if delta_offset_learning:
        for property_name, default_key in delta_offset_property_keys.items():
            prop_keys.setdefault(property_name, default_key)
    restart_h = None
    if transfer_learning and args.restart_from_ckpt_dir is not None:
        restart_h = read_json(Path(args.restart_from_ckpt_dir).absolute().resolve() / 'hyperparameters.json')
        if delta_learning and 'delta_model' not in restart_h:
            raise ValueError('A delta restart checkpoint must contain `delta_model` hyperparameters.')
        if delta_offset_learning:
            if restart_h.get('training_mode') != DELTA_OFFSET_TRAINING_MODE:
                raise ValueError('A delta-offset restart must have `training_mode: delta_offset`.')
            if 'delta_offset_model' not in restart_h:
                raise ValueError('A delta-offset restart is missing `delta_offset_model` hyperparameters.')

    pretrained_ground_ckpt_dir = args.pretrained_ground_ckpt_dir
    if transfer_learning and pretrained_ground_ckpt_dir is None and restart_h is not None:
        model_key = 'delta_model' if delta_learning else 'delta_offset_model'
        pretrained_ground_ckpt_dir = restart_h.get(model_key, {}).get('pretrained_ground_ckpt_dir')
    if transfer_learning and pretrained_ground_ckpt_dir is None:
        flag = '--delta' if delta_learning else '--delta_offset'
        raise ValueError(f'`{flag}` requires `--pretrained_ground_ckpt_dir`.')

    ground_h = None
    ground_params = None
    ground_checkpoint_step = None
    ground_checkpoint_fingerprint = None
    if transfer_learning:
        # Reconstruct architecture from the ground checkpoint instead of duplicating its dimensions in CLI flags.
        pretrained_ground_ckpt_dir = Path(pretrained_ground_ckpt_dir).absolute().resolve().as_posix()
        ground_h = read_json(Path(pretrained_ground_ckpt_dir) / 'hyperparameters.json')
        if 'stack_net' not in ground_h:
            raise ValueError('The pretrained ground checkpoint must contain an ordinary StackNet model.')
        ground_bond_aware = is_bond_aware_stacknet_metadata(ground_h)
        if delta_offset_learning:
            if ground_bond_aware:
                raise ValueError('Delta-offset learning currently requires an ordinary non-bond-aware '
                                 'ground teacher. The student may still be upgraded with `--bond_aware`.')
            if restart_h is None:
                # Load one exact teacher step once. Its parameters remain outside the
                # student optimizer and its immutable identity is saved in metadata.
                ground_params, ground_checkpoint_step, ground_checkpoint_fingerprint = \
                    load_checkpoint_identity(pretrained_ground_ckpt_dir)
            else:
                saved_teacher_h = restart_h['delta_offset_model']
                ground_checkpoint_step = saved_teacher_h.get('ground_checkpoint_step')
                if ground_checkpoint_step is None:
                    raise ValueError('Delta-offset restart metadata is missing `ground_checkpoint_step`.')
                ground_params = load_params_from_ckpt_dir(
                    pretrained_ground_ckpt_dir, step=ground_checkpoint_step)
                ground_checkpoint_fingerprint = checkpoint_fingerprint(
                    pretrained_ground_ckpt_dir,
                    step=ground_checkpoint_step,
                    params=ground_params)
        if restart_h is not None:
            # A restart always reconstructs its saved architecture. CLI flags may not mutate it in place.
            model_key = 'delta_model' if delta_learning else 'delta_offset_model'
            saved_model_h = restart_h[model_key]
            saved_delta_backbone_h = {'stack_net': saved_model_h['backbone']}
            bond_aware = is_bond_aware_stacknet_metadata(saved_delta_backbone_h)
            if args.bond_aware and not bond_aware:
                raise ValueError('`--bond_aware` cannot change a non-bond-aware transfer restart.')
            bond_backbone_upgrade = saved_model_h.get('bond_backbone_upgrade', False)
            bond_descriptor_mode = saved_model_h.get('bond_descriptor_mode', ABSOLUTE_BOND_DESCRIPTOR)
            bond_feature_dim = saved_model_h.get('bond_feature_dim', ABSOLUTE_BOND_FEATURE_DIM)
            bond_parameter_layout = saved_model_h.get('bond_parameter_layout', LEGACY_BOND_LAYOUT)
            if delta_offset_learning:
                saved_step = saved_model_h.get('ground_checkpoint_step')
                saved_fingerprint = saved_model_h.get('ground_checkpoint_fingerprint')
                if (saved_step != ground_checkpoint_step
                        or saved_fingerprint != ground_checkpoint_fingerprint):
                    raise ValueError('The ground teacher no longer matches the immutable identity saved by '
                                     'the delta-offset checkpoint.')
        else:
            # A non-bond ground checkpoint can be upgraded only inside the trainable student.
            bond_backbone_upgrade = bool(args.bond_aware and not ground_bond_aware)
            bond_aware = bool(ground_bond_aware or args.bond_aware)
            if delta_learning:
                # Preserve the established standard-delta behavior exactly.
                bond_descriptor_mode = (RELATIVE_BOND_DESCRIPTOR
                                        if bond_backbone_upgrade else ABSOLUTE_BOND_DESCRIPTOR)
                bond_feature_dim = (RELATIVE_BOND_FEATURE_DIM if bond_backbone_upgrade
                                    else ABSOLUTE_BOND_FEATURE_DIM)
                bond_parameter_layout = (NAMED_BOND_LAYOUT if bond_backbone_upgrade
                                         else LEGACY_BOND_LAYOUT)
            else:
                # Fresh offset runs resolve absolute-active versus relative-to-S0
                # after inspecting the supplied excited-state NPZ schema.
                bond_descriptor_mode = None if bond_backbone_upgrade else ABSOLUTE_BOND_DESCRIPTOR
                bond_feature_dim = None if bond_backbone_upgrade else ABSOLUTE_BOND_FEATURE_DIM
                bond_parameter_layout = NAMED_BOND_LAYOUT if bond_backbone_upgrade else LEGACY_BOND_LAYOUT
    else:
        bond_aware = args.bond_aware
        bond_backbone_upgrade = False
        bond_descriptor_mode = ABSOLUTE_BOND_DESCRIPTOR
        bond_feature_dim = 4
        bond_parameter_layout = LEGACY_BOND_LAYOUT

    def parse_data_file(x):
        if x is not None:
            return Path(x).absolute().resolve().as_posix()
        else:
            return x

    data_file = parse_data_file(args.data_file)
    train_data_file = parse_data_file(args.train_data_file)
    valid_data_file = parse_data_file(args.valid_data_file)

    if data_file is None and (train_data_file is None or valid_data_file is None):
        raise ValueError("Either `--data_file` or (`--train_data_file` + `--valid_data_file`) must be specified.")

    if data_file is None:
        data_files = [train_data_file, valid_data_file]
    else:
        data_files = [data_file]

    offset_state_ids = ()
    if delta_offset_learning:
        observed_state_ids, relative_available, absolute_available = inspect_delta_offset_files(
            data_files=data_files, prop_keys=prop_keys)
        if restart_h is not None:
            offset_state_ids = tuple(restart_h['delta_offset_model'].get('state_ids', (1, 2)))
            if observed_state_ids != offset_state_ids:
                raise ValueError('Delta-offset restart data state IDs must exactly match the saved '
                                 f'{offset_state_ids}; observed {observed_state_ids}.')
            if bond_aware and bond_descriptor_mode == RELATIVE_BOND_DESCRIPTOR and not relative_available:
                raise KeyError('Relative delta-offset restart data is missing required b0/active-state descriptors.')
            if bond_aware and bond_descriptor_mode == ABSOLUTE_BOND_DESCRIPTOR and not absolute_available:
                raise KeyError('Absolute delta-offset restart data requires per-row bond_prob/bond_mask.')
        else:
            offset_state_ids = observed_state_ids
            if bond_backbone_upgrade:
                if relative_available:
                    bond_descriptor_mode = RELATIVE_BOND_DESCRIPTOR
                    bond_feature_dim = RELATIVE_BOND_FEATURE_DIM
                elif absolute_available:
                    bond_descriptor_mode = ABSOLUTE_BOND_DESCRIPTOR
                    bond_feature_dim = ABSOLUTE_BOND_FEATURE_DIM
                else:
                    raise KeyError('Bond-aware delta-offset training requires either active per-row '
                                   'bond_prob/bond_mask or b0 plus active-state bond descriptors.')

    if bond_aware:
        # The first bond-aware port consumes only nonperiodic NPZ files with a fixed graph.
        non_npz_files = [path for path in data_files if Path(path).suffix != '.npz']
        if non_npz_files:
            raise ValueError('`--bond_aware` accepts only NPZ input files with precomputed graph arrays.')
        if args.mic:
            raise ValueError('`--bond_aware` currently supports only nonperiodic precomputed graphs.')
        if args.restart_from_ckpt_dir is not None and not transfer_learning:
            raise ValueError('Bond-aware SO3krates models must be trained from fresh initialization.')

    shift_by = args.shift_by
    shifts = args.shifts

    if shifts is not None:
        shifts = {int(k): float(v) for (k, v) in shifts.items()}

    if args.ckpt_dir is None:
        ckpt_dir = (Path(os.getcwd()).absolute().resolve() / 'module').as_posix()
    else:
        ckpt_dir = (Path(args.ckpt_dir).absolute().resolve()).as_posix()

    restart_from_ckpt_dir = None
    if args.restart_from_ckpt_dir is not None:
        restart_from_ckpt_dir = (Path(args.restart_from_ckpt_dir).absolute().resolve()).as_posix()
        assert restart_from_ckpt_dir != ckpt_dir

    if Path(ckpt_dir).exists():
        raise FileExistsError(f'Checkpoint directory {ckpt_dir} already exists.')

    if transfer_learning:
        # Use checkpoint geometry settings for both neighbor validation and the transferred representation.
        ground_geometry_h = next(
            embedding_h['geometry_embed']
            for embedding_h in ground_h['stack_net']['geometry_embeddings']
            if 'geometry_embed' in embedding_h)
        r_cut = float(ground_geometry_h['r_cut'])
        mic = bool(ground_geometry_h.get('mic', False))
    else:
        r_cut = args.r_cut
        mic = args.mic

    if bond_aware and mic:
        raise ValueError('Bond-aware transfer learning currently supports only nonperiodic pretrained checkpoints.')

    F = args.F
    L = args.L
    degrees = args.degrees

    eval_every_t = args.eval_every_t
    use_wandb = args.use_wandb

    lr = args.lr
    lr_decay_plateau = args.lr_decay_plateau
    if lr_decay_plateau is not None:
        lr_decay_plateau = {k: float(v) for k, v in lr_decay_plateau.items()}

    lr_decay_exp = {'exponential': args.lr_decay_exp} if args.lr_decay_exp is not None else args.lr_decay_exp
    lr_warmup = args.lr_warmup

    clip_by_global_norm = args.clip_by_global_norm

    epochs = args.epochs
    steps = args.steps
    lr_stop = args.lr_stop

    inputs = list(args.inputs)
    targets = list(args.targets)
    if delta_learning:
        # Delta optimization always supervises both energy and gradient-derived force corrections.
        targets = [pn.delta_energy_1, pn.delta_energy_2,
                   pn.delta_force_1, pn.delta_force_2]
    elif delta_offset_learning:
        targets = [pn.offset_energy, pn.offset_force]
        if pn.active_state not in inputs:
            inputs.append(pn.active_state)

    if bond_aware:
        # Ensure Coach/DataTuple preserve the fixed edge mask and both bond descriptor tensors.
        for bond_input in (pn.pair_mask, pn.bond_prob, pn.bond_mask):
            if bond_input not in inputs:
                inputs.append(bond_input)

    if delta_learning and bond_aware:
        # Preserve every state descriptor needed for the two shared-backbone passes and later ground reconstruction.
        state_bond_inputs = (pn.bond_prob_s0, pn.bond_mask_s0,
                             pn.bond_prob_s1, pn.bond_mask_s1,
                             pn.bond_prob_s2, pn.bond_mask_s2)
        for state_bond_input in state_bond_inputs:
            if state_bond_input not in inputs:
                inputs.append(state_bond_input)
    elif delta_offset_learning and bond_aware and bond_descriptor_mode == RELATIVE_BOND_DESCRIPTOR:
        # S1-only datasets need only b0+b1; two-state datasets additionally retain b2.
        state_bond_inputs = [pn.bond_prob_s0, pn.bond_mask_s0]
        for state in offset_state_ids:
            state_bond_inputs.extend((getattr(pn, f'bond_prob_s{state}'),
                                      getattr(pn, f'bond_mask_s{state}')))
        for state_bond_input in state_bond_inputs:
            if state_bond_input not in inputs:
                inputs.append(state_bond_input)

    if mic:
        inputs += [pn.unit_cell]
        inputs += [pn.cell_offset]
        if delta_offset_learning:
            # DataSet needs PBC flags to generate periodic neighbors even though
            # the network itself consumes only cell offsets.
            inputs += [pn.pbc]

    _loss_weights = args.loss_weights
    missing_loss_weights = [target for target in targets if target not in _loss_weights]
    if missing_loss_weights:
        raise ValueError(f'No loss weights were provided for targets {missing_loss_weights}.')
    loss_weights = {k: float(v) for (k, v) in _loss_weights.items() if k in targets}

    total_loss_weight = sum([x for x in loss_weights.values()])
    if total_loss_weight <= 0:
        raise ValueError('The selected loss weights must sum to a positive value.')
    effective_loss_weights = {k: v / total_loss_weight for k, v in loss_weights.items()}

    n_train = args.n_train
    n_valid = args.n_valid
    n_test = args.n_test

    if data_file is not None:
        if n_train is None or n_valid is None:
            raise ValueError('If only a single `--data_file` is provided, please specify the number of training'
                             'and validation samples via `--n_train` and `--n_valid`.')

    model_seed = args.model_seed
    training_seed = args.training_seed
    data_seed = args.data_seed

    units = args.units
    conversion_table = {}
    if units is not None:
        for (q, v) in units.items():
            k = prop_keys[q]
            conversion_table[k] = eval(v)

    all_data = []
    graph_metadata = []
    for d in data_files:
        extension = os.path.splitext(d)[1]
        if delta_offset_learning:
            data, file_state_ids = load_delta_offset_npz(
                path=d,
                inputs=inputs,
                prop_keys=prop_keys,
                conversion_table=conversion_table,
                bond_descriptor_mode=bond_descriptor_mode)
            if not set(file_state_ids).issubset(offset_state_ids):
                raise ValueError(f'Unexpected active states {file_state_ids} in {d}.')
            if bond_aware:
                graph_metadata.append(load_precomputed_graph_metadata(d, r_cut=r_cut))
        elif extension == '.npz':
            # Disable object loading so the graph contract remains numeric and inspectable.
            data = dict(np.load(d, allow_pickle=False))
            if bond_aware:
                graph_metadata.append(load_precomputed_graph_metadata(d, r_cut=r_cut))
        else:
            load_stress = pn.stress in targets
            data_loader = AseDataLoader(d, load_stress=load_stress, neighbors_format='dense')
            data = data_loader.load_all()

        if not delta_offset_learning:
            data = unit_convert_data(data, table=conversion_table)
        if pn.stress in targets:
            cell_key = prop_keys[pn.unit_cell]
            stress_key = prop_keys[pn.stress]

            stress = data[stress_key]
            try:
                assert stress.shape[-2:] == (3, 3)
            except AssertionError:
                raise ValueError('Stress tensor must be a matrix with shape (3,3). '
                                 'Voigt convention not supported yet.')

            # re-scale stress with cell volume
            cells = data[cell_key]  # shape: (B,3,3)
            cell_volumes = np.abs(np.linalg.det(cells))  # shape: (B)
            data[stress_key] = stress * cell_volumes[:, None, None]

        # Keep only declared model inputs and targets before DataSet can reshape unrelated per-frame metadata.
        # This is also important when an ordinary ground model consumes a graph-rich NPZ such as I02_s0.npz.
        if not delta_offset_learning:
            data = select_data_for_model(data=data,
                                         inputs=inputs,
                                         targets=targets,
                                         prop_keys=prop_keys)
        all_data += [data]

    if len(all_data) == 2:
        n_train = len(all_data[0][prop_keys[pn.atomic_position]])
        n_valid = len(all_data[1][prop_keys[pn.atomic_position]])

        data = jax.tree_util.tree_map(lambda x, y: np.concatenate([x, y]), *all_data)

        # Reuse available sidecar tolerance metadata after both fixed-graph datasets are concatenated.
        combined_metadata = next((metadata for metadata in graph_metadata if metadata is not None), None)
        data_set = DataSet(data=data, prop_keys=prop_keys, graph_metadata=combined_metadata)
        data_set.index_split(data_idx_train=list(range(n_train)),
                             data_idx_valid=list(range(n_train, int(n_train+n_valid))),
                             data_idx_test=[],
                             r_cut=r_cut,
                             training=True,
                             mic=mic,
                             precomputed_graph=bond_aware)
    elif len(all_data) == 1:
        data = all_data[0]
        # Pass adjacent metadata into the dataset validator when it is available.
        combined_metadata = next((metadata for metadata in graph_metadata if metadata is not None), None)
        data_set = DataSet(data=data, prop_keys=prop_keys, graph_metadata=combined_metadata)
        if delta_offset_learning:
            idx_train, idx_valid, idx_test = generate_grouped_split_indices(
                group_ids=data_set.data[DELTA_OFFSET_GROUP_KEY],
                n_train=n_train,
                n_valid=n_valid,
                n_test=n_test,
                seed=data_seed)
            data_set.index_split(data_idx_train=idx_train,
                                 data_idx_valid=idx_valid,
                                 data_idx_test=idx_test,
                                 r_cut=r_cut,
                                 training=True,
                                 mic=mic,
                                 precomputed_graph=bond_aware)
        else:
            data_set.random_split(n_train=n_train,
                                  n_valid=n_valid,
                                  n_test=n_test,
                                  r_cut=r_cut,
                                  training=True,
                                  mic=mic,
                                  seed=data_seed,
                                  precomputed_graph=bond_aware)
    else:
        raise RuntimeError('You should not end up here. Please file an issue :-)')

    if not transfer_learning:
        # Ground-state training retains its existing energy-shift behavior.
        if shift_by == 'mean':
            data_set.shift_x_by_mean_x(x=pn.energy)
        elif shift_by == 'atomic_number':
            data_set.shift_x_by_type(x=pn.energy, shifts=shifts)
        elif shift_by == 'lse':
            data_set.shift_x_by_type(x=pn.energy)

    d = data_set.get_data_split()
    if delta_offset_learning:
        train_states = set(int(state) for state in np.unique(
            d['train'][prop_keys[pn.active_state]]))
        missing_train_states = sorted(set(offset_state_ids) - train_states)
        if missing_train_states:
            raise ValueError('Grouped training split contains no samples for active states '
                             f'{missing_train_states}; adjust n_train or data_seed.')
        teacher_net = init_stack_net(ground_h)
        teacher_net.reset_prop_keys(prop_keys=prop_keys)
        teacher_scales = read_json(Path(pretrained_ground_ckpt_dir) / 'scales.json')
        teacher_batch_size = args.batch_size if args.batch_size is not None else 32
        generate_delta_offset_targets(
            # Coach optimizes/evaluates only train+valid. A non-precomputed
            # DataSet intentionally does not build test neighbors in training
            # mode; the standalone evaluator reconstructs them when requested.
            data_split={name: d[name] for name in ('train', 'valid')},
            teacher_net=teacher_net,
            teacher_params=ground_params,
            teacher_scales=teacher_scales,
            prop_keys=prop_keys,
            batch_size=teacher_batch_size)

    scales = {}
    if args.loss_variance_scaling:
        for t in targets:
            if t == pn.stress:
                scales[prop_keys[t]] = 1 / np.nanvar(d['train'][prop_keys[t]], axis=0)
            elif t in (pn.energy, pn.delta_energy_1, pn.delta_energy_2, pn.offset_energy):
                scales[prop_keys[t]] = 1 / np.nanvar(d['train'][prop_keys[t]])
            elif t in (pn.force, pn.delta_force_1, pn.delta_force_2, pn.offset_force):
                force_data_train = d['train'][prop_keys[t]]
                node_msk_train = d['train'][prop_keys[pn.node_mask]]
                scales[prop_keys[t]] = 1 / np.nanvar(force_data_train[node_msk_train])
            else:
                raise NotImplementedError('Loss with variance scaling currently only implemented for loss with '
                                          'energy, delta energy, forces, delta forces, and/or stress.')
    else:
        scales = None

    n_heads = args.H

    so3krates_layer_kwargs = {'degrees': degrees,
                              'n_heads': n_heads}

    if args.so3krates_layer_kwargs is not None:
        so3krates_layer_kwargs.update(args.so3krates_layer_kwargs)

    # Make the public flag authoritative even when a JSON kwargs dictionary also contains the field.
    so3krates_layer_kwargs['bond_aware'] = bond_aware

    if transfer_learning and args.zbl_repulsion:
        raise ValueError('ZBL repulsion belongs to the ground energy head and is not a transfer-head option.')
    if args.zbl_repulsion:
        print('Running with ZBL repulsion.')

    geometry_embed_kwargs = {'degrees': degrees,
                             'mic': mic,
                             'r_cut': r_cut}
    if args.geometry_embed_kwargs is not None:
        geometry_embed_kwargs.update(args.geometry_embed_kwargs)

    if delta_learning:
        if restart_h is not None:
            # Preserve the saved upgraded/legacy backbone layout and descriptor contract exactly.
            net = init_delta_model(restart_h,
                                   freeze_pretrained_backbone=args.freeze_pretrained_backbone)
            net.reset_prop_keys(prop_keys=prop_keys)
        else:
            # Reuse the exact ground representation, optionally adding versioned relative-bond branches to its clone.
            delta_backbone_h = (upgrade_stacknet_for_relative_bond_delta(ground_h)
                                if bond_backbone_upgrade else ground_h)
            delta_backbone = init_stack_net(delta_backbone_h)
            delta_backbone.reset_prop_keys(prop_keys=prop_keys)
            net = init_state_specific_delta_so3krates(
                backbone=delta_backbone,
                pretrained_ground_ckpt_dir=pretrained_ground_ckpt_dir,
                freeze_pretrained_backbone=args.freeze_pretrained_backbone,
                bond_backbone_upgrade=bond_backbone_upgrade,
                bond_descriptor_mode=bond_descriptor_mode,
                bond_feature_dim=bond_feature_dim,
                bond_parameter_layout=bond_parameter_layout)
        obs_fn = get_delta_energy_force_fn(net)
    elif delta_offset_learning:
        if restart_h is not None:
            net = init_delta_offset_model(
                restart_h,
                freeze_pretrained_backbone=args.freeze_pretrained_backbone)
            net = net.clone(
                pretrained_ground_ckpt_dir=pretrained_ground_ckpt_dir,
                ground_checkpoint_step=ground_checkpoint_step,
                ground_checkpoint_fingerprint=ground_checkpoint_fingerprint)
            net.reset_prop_keys(prop_keys=prop_keys)
        else:
            offset_backbone_h = (
                upgrade_stacknet_for_bond_delta(
                    ground_h=ground_h,
                    bond_feature_dim=bond_feature_dim,
                    bond_parameter_layout=bond_parameter_layout)
                if bond_backbone_upgrade else ground_h)
            offset_backbone = init_stack_net(offset_backbone_h)
            offset_backbone.reset_prop_keys(prop_keys=prop_keys)
            required_source_properties = [
                pn.atomic_position, pn.atomic_type, pn.energy, pn.force, pn.active_state]
            if mic:
                required_source_properties.extend((pn.unit_cell, pn.pbc))
            if bond_aware:
                required_source_properties.extend((pn.idx_i, pn.idx_j, pn.pair_mask))
                if bond_descriptor_mode == ABSOLUTE_BOND_DESCRIPTOR:
                    required_source_properties.extend((pn.bond_prob, pn.bond_mask))
                else:
                    required_source_properties.extend((pn.bond_prob_s0, pn.bond_mask_s0))
                    for state in offset_state_ids:
                        required_source_properties.extend(
                            (getattr(pn, f'bond_prob_s{state}'),
                             getattr(pn, f'bond_mask_s{state}')))
            required_dataset_keys = tuple(
                prop_keys[name] for name in dict.fromkeys(required_source_properties))
            net = init_state_routed_offset_so3krates(
                backbone=offset_backbone,
                pretrained_ground_ckpt_dir=pretrained_ground_ckpt_dir,
                ground_checkpoint_step=ground_checkpoint_step,
                ground_checkpoint_fingerprint=ground_checkpoint_fingerprint,
                freeze_pretrained_backbone=args.freeze_pretrained_backbone,
                bond_backbone_upgrade=bond_backbone_upgrade,
                bond_descriptor_mode=bond_descriptor_mode,
                bond_feature_dim=bond_feature_dim,
                bond_parameter_layout=bond_parameter_layout,
                state_ids=offset_state_ids,
                required_dataset_keys=required_dataset_keys)
        obs_fn = get_delta_offset_energy_force_fn(net)
    else:
        obs = [Energy(prop_keys=prop_keys, zbl_repulsion=args.zbl_repulsion)]
        net = So3krates(prop_keys=prop_keys,
                        F=F,
                        n_layer=L,
                        obs=obs,
                        geometry_embed_kwargs=geometry_embed_kwargs,
                        so3krates_layer_kwargs=so3krates_layer_kwargs)

    if not transfer_learning and pn.force in targets:
        if pn.stress in targets:
            obs_fn = get_energy_force_stress_fn(net)
        else:
            obs_fn = get_obs_and_force_fn(net)
    elif not transfer_learning:
        obs_fn = get_observable_fn(net)

    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

    opt = Optimizer(clip_by_global_norm=clip_by_global_norm)

    tx = opt.get(learning_rate=lr)

    def autoset_batch_size(u):
        if u < 500:
            return 1
        elif 500 <= u < 1000:
            return 5
        elif 1000 <= u < 10_000:
            return 10
        elif u >= 10_000:
            return 100

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = autoset_batch_size(n_train)

    training_batch_size = batch_size if args.training_batch_size is None else args.training_batch_size
    validation_batch_size = batch_size if args.validation_batch_size is None else args.validation_batch_size

    if epochs is None and steps is None:
        assert lr_stop is not None
        if args.lr_decay_exp is None and lr_decay_plateau is None:
            raise ValueError('No learning rate decay is specified. At the same time neither epochs nor steps is speci'
                             f'fied such that the training is stopped when the learning rate is below {lr_stop}. Thus'
                             f'either specify a learning rate decay using the `--lr_decay_exp` or the'
                             f' `lr_decay_plateau` argument. Alternatively, specify a number of epochs, steps.')
        _epochs = 1_000_000_000
    elif epochs is None and steps is not None:
        _epochs = int(steps / (n_train / training_batch_size))
    elif epochs is not None and steps is None:
        _epochs = epochs
    elif epochs is not None and steps is not None:
        raise ValueError('Only epochs or steps argument can be specified.')
    else:
        msg = 'One should not end up here. This is likely due to a bug in the mlff package. Please report to ' \
              'https://github.com/thorben-frank/mlff'
        raise RuntimeError(msg)

    coach = Coach(inputs=inputs,
                  targets=targets,
                  epochs=_epochs,
                  training_batch_size=training_batch_size,
                  validation_batch_size=validation_batch_size,
                  loss_weights=effective_loss_weights,
                  ckpt_dir=ckpt_dir,
                  data_path=data_file,
                  train_data_path=data_file if train_data_file is None else train_data_file,
                  valid_data_path=data_file if valid_data_file is None else valid_data_file,
                  net_seed=model_seed,
                  training_seed=training_seed,
                  stop_lr_min=lr_stop)

    data_tuple = DataTuple(inputs=inputs,
                           targets=targets,
                           prop_keys=prop_keys)

    train_ds = data_tuple(d['train'])
    valid_ds = data_tuple(d['valid'])

    inputs = jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
    transferred_backbone_paths = ()
    if restart_from_ckpt_dir is None:
        # Initialize the complete delta tree first so its new state head receives independent random parameters.
        params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
        if transfer_learning:
            if ground_params is None:
                ground_params = load_params_from_ckpt_dir(pretrained_ground_ckpt_dir)
            params, transferred_backbone_paths = load_pretrained_backbone(
                params,
                ground_params,
                strict=True,
                allow_bond_upgrade=bond_backbone_upgrade,
                return_transferred_paths=True)
    else:
        print(f"Restarting training from {restart_from_ckpt_dir}.")
        params = load_params_from_ckpt_dir(restart_from_ckpt_dir)
        if transfer_learning and args.freeze_pretrained_backbone:
            if ground_params is None:
                ground_params = load_params_from_ckpt_dir(pretrained_ground_ckpt_dir)
            transferred_backbone_paths = get_pretrained_backbone_paths(
                delta_variables=params,
                ground_variables=ground_params,
                allow_bond_upgrade=bond_backbone_upgrade)

    frozen_param_paths = None
    if transfer_learning and args.freeze_pretrained_backbone:
        # Exact leaf paths freeze only the transferred representation. New bond branches and the head remain trainable.
        frozen_param_paths = tuple(('params', 'backbone', *path) for path in transferred_backbone_paths)
        if not frozen_param_paths:
            raise ValueError('No transferred backbone parameter paths were found to freeze.')
    loss_fn = get_loss_fn(obs_fn=obs_fn,
                          weights=effective_loss_weights,
                          scales=scales,
                          prop_keys=prop_keys,
                          frozen_param_paths=frozen_param_paths)

    train_state, h_train_state = create_train_state(net,
                                                    params,
                                                    tx,
                                                    polyak_step_size=None,
                                                    plateau_lr_decay=lr_decay_plateau,
                                                    scheduled_lr_decay=lr_decay_exp,
                                                    lr_warmup=lr_warmup
                                                    )

    h_net = net.__dict_repr__()
    h_opt = opt.__dict_repr__()
    h_coach = coach.__dict_repr__()
    h_dataset = data_set.__dict_repr__()
    h_mode = ({'training_mode': DELTA_OFFSET_TRAINING_MODE}
              if delta_offset_learning else {})
    h = bundle_dicts([h_mode, h_net, h_opt, h_coach, h_dataset, h_train_state])

    Path(ckpt_dir).mkdir(parents=True, exist_ok=False)
    if data_file is not None:
        data_set.save_splits_to_file(ckpt_dir, 'splits.json')

    data_set.save_scales(ckpt_dir, 'scales.json')
    save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

    if use_wandb:
        wandb.init(config=h, **args.wandb_init)

    coach.run(train_state=train_state,
              train_ds=train_ds,
              valid_ds=valid_ds,
              loss_fn=loss_fn,
              eval_every_t=eval_every_t,
              log_every_t=1,
              ckpt_manager_options=args.ckpt_manager_options,
              restart_by_nan=True,
              use_wandb=use_wandb)


if __name__ == '__main__':
    train_so3krates()
