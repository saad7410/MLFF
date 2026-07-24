"""Evaluation and reconstruction for SO3krates delta-offset checkpoints.

This path intentionally consumes only the active-state ``E``/``F`` labels.  A
frozen ground-state teacher supplies the otherwise unavailable baseline on
excited-state geometries.
"""

import argparse
import json
import os

from pathlib import Path
from pprint import pprint
from typing import Dict, Iterable, Mapping, Sequence

import jax
import numpy as np
from ase.units import *  # noqa: F403 - retain the existing evaluator's unit CLI

from mlff.cAPI.delta_mlff_eval import _geometry_settings
from mlff.cAPI.mlff_eval import is_bond_aware_stacknet_metadata, unit_convert_data
from mlff.cAPI.process_argparse import StoreDictKeyPair
from mlff.data import DataSet, DataTuple, load_precomputed_graph_metadata
from mlff.io import checkpoint_fingerprint, load_params_from_ckpt_dir, read_json
from mlff.nn import init_delta_offset_model, restore_ground_prediction_units
from mlff.nn.stacknet import get_delta_offset_energy_force_fn, get_obs_and_force_fn, init_stack_net
from mlff.properties import property_names as pn
from mlff.training import Coach


SOURCE_INDEX_KEY = '__delta_offset_source_index__'
OFFSET_SIGN_CONVENTION = 'active_minus_teacher'


def _require_delta_offset_checkpoint(h: Mapping, ckpt_dir) -> Mapping:
    """Return mode metadata, rejecting ordinary and physical-delta checkpoints."""
    if h.get('training_mode') != 'delta_offset' or 'delta_offset_model' not in h:
        raise ValueError(f'{ckpt_dir} is not a delta-offset checkpoint '
                         '(expected `training_mode: delta_offset`).')
    metadata = h['delta_offset_model']
    if metadata.get('offset_sign_convention') != OFFSET_SIGN_CONVENTION:
        raise ValueError('Unsupported or missing delta-offset sign convention; expected '
                         f'`{OFFSET_SIGN_CONVENTION}`.')
    expected_normalization = {'target_shift': 'none', 'target_scale': 1.0}
    if metadata.get('normalization') != expected_normalization:
        raise ValueError('Delta-offset evaluation supports only raw-unit targets with normalization '
                         f'{expected_normalization}.')
    if metadata.get('teacher_prediction_mode') != 'online_before_optimization':
        raise ValueError('Delta-offset checkpoint does not record the supported frozen-teacher '
                         'prediction mode.')
    return metadata


def _state_vector(values, name='active_state') -> np.ndarray:
    """Normalize a structure-scalar state array without guessing its index base."""
    values = np.asarray(values)
    if values.ndim == 0 or (values.ndim > 1 and np.prod(values.shape[1:]) != 1):
        raise ValueError(f'`{name}` must have shape (B,) or (B, 1).')
    values = values.reshape(-1)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f'`{name}` must contain integer state IDs.')
    return values.astype(np.int32)


def _validate_and_select_excited_states(values,
                                        trained_states: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Validate zero-based labels and return the excited-state selection mask."""
    states = _state_vector(values)
    invalid = sorted(set(np.unique(states).tolist()) - {0, 1, 2})
    if invalid:
        raise ValueError(f'Active states must be zero-based values in {{0, 1, 2}}; found {invalid}.')

    trained_states = tuple(sorted({int(state) for state in trained_states}))
    if not trained_states or any(state not in (1, 2) for state in trained_states):
        raise ValueError('Delta-offset checkpoint `state_ids` must be a non-empty subset of {1, 2}.')
    unsupported = sorted(set(np.unique(states[states > 0]).tolist()) - set(trained_states))
    if unsupported:
        raise ValueError(f'Dataset contains excited states not supported by this checkpoint: {unsupported}.')

    excited = states > 0
    if not excited.any():
        raise ValueError('Delta-offset evaluation found no excited-state rows after excluding state 0.')
    return states, excited


def _load_excited_npz_data(path,
                           conversion_table: Mapping,
                           inputs: Sequence[str],
                           prop_keys: Mapping[str, str],
                           trained_states: Sequence[int],
                           bond_aware: bool,
                           r_cut: float,
                           mic: bool = False,
                           bond_descriptor_mode: str = 'absolute_state'):
    """Load only model inputs and active labels, explicitly ignoring oracle-like arrays."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Delta-offset evaluation requires an NPZ with active-state E/F labels.')

    # Ordinary DataSet splitting synthesizes neighbor indices, node masks, and periodic edge offsets.
    # A fixed bond-aware graph, in contrast, must preserve every serialized model input.
    derived_properties = {pn.idx_i, pn.idx_j, pn.node_mask, pn.cell_offset}
    raw_inputs = tuple(
        (name for name in inputs if name != pn.node_mask)
        if bond_aware else
        (name for name in inputs if name not in derived_properties))
    required_properties = list(dict.fromkeys((*raw_inputs, pn.energy, pn.force, pn.active_state)))
    if mic and not bond_aware:
        for periodic_property in (pn.unit_cell, pn.pbc):
            if periodic_property not in required_properties:
                required_properties.append(periodic_property)
    required_properties = tuple(required_properties)
    missing_mappings = [name for name in required_properties if name not in prop_keys]
    if missing_mappings:
        raise KeyError(f'Missing property mappings for delta-offset quantities {missing_mappings}.')
    required_keys = tuple(dict.fromkeys(prop_keys[name] for name in required_properties))

    # Access only declared arrays. In particular, energy_all/forces_all are neither read nor trusted.
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        source_keys = {key: key for key in required_keys}
        if bond_descriptor_mode == 'relative_to_s0':
            # Match training: relative mode always canonicalizes the fixed-graph
            # bond arrays from b0. A source file may also contain per-row active
            # canonical arrays, which must not be mistaken for the S0 baseline.
            aliases = {
                prop_keys[pn.bond_prob]: prop_keys[pn.bond_prob_s0],
                prop_keys[pn.bond_mask]: prop_keys[pn.bond_mask_s0],
            }
            for target_key, source_key in aliases.items():
                if target_key in source_keys:
                    source_keys[target_key] = source_key
        missing_keys = [target for target, source in source_keys.items()
                        if source not in available]
        if missing_keys:
            raise KeyError(f'Missing delta-offset input/label arrays {missing_keys}.')
        data = {target: np.asarray(archive[source])
                for target, source in source_keys.items()}
        # Preserve an explicitly supplied node mask; otherwise DataSet derives it
        # from padded atom types for both generated and fixed-graph paths.
        node_mask_key = prop_keys.get(pn.node_mask)
        if node_mask_key in archive.files:
            data[node_mask_key] = np.asarray(archive[node_mask_key])

    state_key = prop_keys[pn.active_state]
    states, excited = _validate_and_select_excited_states(
        data[state_key],
        trained_states=trained_states)
    n_rows = len(states)

    position_key = prop_keys[pn.atomic_position]
    atomic_type_key = prop_keys[pn.atomic_type]
    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    positions = np.asarray(data[position_key])
    atomic_types = np.asarray(data[atomic_type_key])
    active_energy = np.asarray(data[energy_key])
    active_force = np.asarray(data[force_key])
    if positions.ndim != 3 or positions.shape[0] != n_rows or positions.shape[-1] != 3:
        raise ValueError('Atomic positions must have shape (B, N, 3) aligned with active states.')
    if active_energy.shape not in ((n_rows,), (n_rows, 1)):
        raise ValueError('Active-state energy labels must have shape (B,) or (B, 1).')
    if active_force.shape != positions.shape:
        raise ValueError('Active-state force labels must have shape (B, N, 3) matching positions.')
    if atomic_types.ndim == 2:
        if atomic_types.shape != positions.shape[:2]:
            raise ValueError('Batched atomic types must have shape (B, N) matching positions.')
    elif atomic_types.ndim != 1 or atomic_types.shape[0] != positions.shape[1]:
        raise ValueError('Atomic types must have shape (B, N) or a shared shape (N,).')

    selected = {}
    for key, value in data.items():
        if value.ndim > 0 and value.shape[0] == n_rows:
            selected[key] = value[excited]
        else:
            # Preserve structure-independent atom types/graph templates for DataSet's established broadcasting.
            selected[key] = value
    selected[state_key] = states[excited, None]
    selected[SOURCE_INDEX_KEY] = np.flatnonzero(excited)[:, None]
    selected = unit_convert_data(selected, table=dict(conversion_table))

    if not np.isfinite(selected[energy_key]).all():
        raise ValueError('Active energies must be finite on every selected excited-state row.')
    if prop_keys.get(pn.node_mask) in selected:
        real_atoms = np.asarray(selected[prop_keys[pn.node_mask]]).astype(bool)
    else:
        selected_z = np.asarray(selected[atomic_type_key])
        real_atoms = selected_z != 0
        if selected_z.ndim == 1:
            real_atoms = np.broadcast_to(real_atoms, selected[force_key].shape[:2])
    invalid_real_force = (
        ~np.isfinite(selected[force_key])
        & np.broadcast_to(real_atoms[..., None], selected[force_key].shape))
    if invalid_real_force.any():
        raise ValueError('Active forces must be finite on every non-padded atom.')

    graph_metadata = (load_precomputed_graph_metadata(path, r_cut=r_cut)
                      if bond_aware else None)
    return selected, graph_metadata


def _evaluation_split(data_set,
                      ckpt_dir,
                      evaluate_on,
                      from_split,
                      n_test,
                      r_cut,
                      mic,
                      bond_aware):
    if from_split is not None:
        split_counts = {'train': (None, 0, 0),
                        'valid': (0, None, 0),
                        'test': (0, 0, n_test)}
        n_train, n_valid, resolved_n_test = split_counts[evaluate_on]
        data_set.load_split(file=Path(ckpt_dir) / 'splits.json',
                            n_train=n_train,
                            n_valid=n_valid,
                            n_test=resolved_n_test,
                            r_cut=r_cut,
                            mic=mic,
                            split_name=from_split,
                            precomputed_graph=bond_aware)
        return data_set.get_data_split()[evaluate_on]

    n_apply = data_set.n_data if n_test is None else n_test
    if n_apply < 0 or n_apply > data_set.n_data:
        raise ValueError(f'`--n_test` must select between 0 and {data_set.n_data} excited structures.')
    split_indices = {'train': np.asarray([], dtype=int),
                     'valid': np.asarray([], dtype=int),
                     'test': np.asarray([], dtype=int)}
    split_indices[evaluate_on] = np.arange(n_apply)
    data_set.index_split(data_idx_train=split_indices['train'],
                         data_idx_valid=split_indices['valid'],
                         data_idx_test=split_indices['test'],
                         r_cut=r_cut,
                         mic=mic,
                         training=False,
                         precomputed_graph=bond_aware)
    return data_set.get_data_split()[evaluate_on]


def _batched_predictions(params, obs_fn, inputs: Mapping, batch_size: int) -> Dict:
    """Evaluate every structure, including a final partial batch."""
    if batch_size <= 0:
        raise ValueError('`--batch_size` must be positive.')
    if not inputs:
        raise ValueError('Delta-offset evaluation inputs are empty.')
    n_data = len(next(iter(inputs.values())))
    if n_data == 0:
        raise ValueError('Delta-offset evaluation selection is empty.')

    chunks = {}
    for start in range(0, n_data, batch_size):
        stop = min(start + batch_size, n_data)
        batch = jax.tree_util.tree_map(lambda value: value[start:stop], inputs)
        prediction = jax.device_get(obs_fn(params, batch))
        if not chunks:
            chunks = {key: [np.asarray(value)] for key, value in prediction.items()}
        else:
            if set(prediction) != set(chunks):
                raise ValueError('Delta-offset observable returned inconsistent keys between batches.')
            for key, value in prediction.items():
                chunks[key].append(np.asarray(value))
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _offset_evaluation_arrays(teacher_energy,
                              teacher_force,
                              predicted_offset_energy,
                              predicted_offset_force,
                              target_energy,
                              target_force) -> Dict[str, np.ndarray]:
    """Construct teacher-relative targets and reconstructed predictions in raw units."""
    teacher_energy = np.asarray(teacher_energy)
    teacher_force = np.asarray(teacher_force)
    predicted_offset_energy = np.asarray(predicted_offset_energy)
    predicted_offset_force = np.asarray(predicted_offset_force)
    target_energy = np.asarray(target_energy)
    target_force = np.asarray(target_force)

    if teacher_energy.shape != target_energy.shape:
        raise ValueError('Teacher and active-state energy arrays must have identical shapes.')
    if teacher_force.shape != target_force.shape:
        raise ValueError('Teacher and active-state force arrays must have identical shapes.')
    if predicted_offset_energy.shape != target_energy.shape:
        raise ValueError('Predicted offset and active-state energy arrays must have identical shapes.')
    if predicted_offset_force.shape != target_force.shape:
        raise ValueError('Predicted offset and active-state force arrays must have identical shapes.')

    target_offset_energy = target_energy - teacher_energy
    target_offset_force = target_force - teacher_force
    return {
        'teacher_energy': teacher_energy,
        'teacher_force': teacher_force,
        'target_offset_energy': target_offset_energy,
        'target_offset_force': target_offset_force,
        'prediction_offset_energy': predicted_offset_energy,
        'prediction_offset_force': predicted_offset_force,
        'target_reconstructed_energy': target_energy,
        'target_reconstructed_force': target_force,
        'prediction_reconstructed_energy': teacher_energy + predicted_offset_energy,
        'prediction_reconstructed_force': teacher_force + predicted_offset_force,
    }


def _metric_triplet(prediction, target) -> Dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    error = prediction - target
    with np.errstate(divide='ignore', invalid='ignore'):
        # Match the repository evaluator's definitions without dispatching tiny
        # metric arrays through a JAX accelerator.
        return {'mae': float(np.mean(np.abs(error))),
                'rmse': float(np.sqrt(np.mean(error ** 2))),
                'R2': float(1 - (np.std(error) / np.std(target)) ** 2)}


def _quantity_metrics(arrays: Mapping[str, np.ndarray],
                      row_mask: np.ndarray,
                      node_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    pairs = {
        'offset_energy': ('prediction_offset_energy', 'target_offset_energy'),
        'offset_force': ('prediction_offset_force', 'target_offset_force'),
        'reconstructed_energy': ('prediction_reconstructed_energy',
                                 'target_reconstructed_energy'),
        'reconstructed_force': ('prediction_reconstructed_force',
                                'target_reconstructed_force'),
    }
    metrics = {}
    for quantity, (prediction_key, target_key) in pairs.items():
        prediction = np.asarray(arrays[prediction_key])[row_mask]
        target = np.asarray(arrays[target_key])[row_mask]
        valid = np.isfinite(target)
        if quantity.endswith('_force'):
            valid &= np.broadcast_to(node_mask[row_mask, ..., None], target.shape)
        metrics[quantity] = _metric_triplet(prediction[valid], target[valid])
    return metrics


def _delta_offset_metrics(arrays: Mapping[str, np.ndarray],
                          active_state,
                          node_mask,
                          trained_states: Sequence[int]) -> Dict:
    states = _state_vector(active_state)
    node_mask = np.asarray(node_mask).astype(bool)
    if node_mask.shape[0] != len(states):
        raise ValueError('Node masks and active-state labels must share the batch dimension.')

    metrics = {}
    for state in trained_states:
        row_mask = states == state
        if not row_mask.any():
            continue
        state_metrics = _quantity_metrics(arrays=arrays,
                                          row_mask=row_mask,
                                          node_mask=node_mask)
        metrics[f'state_{state}'] = {
            'n_structures': int(row_mask.sum()),
            **state_metrics,
        }
    return metrics


def _resolve_data_path(args, coach):
    apply_to = args.apply_to
    from_split = args.from_split
    if apply_to is None and coach.data_path is not None:
        apply_to = coach.data_path
        from_split = 'split' if from_split is None else from_split
    elif apply_to is None:
        apply_to = coach.train_data_path if args.on == 'train' else coach.valid_data_path
    if apply_to is None:
        raise ValueError('No evaluation data path was supplied or stored in the delta-offset checkpoint.')
    return apply_to, from_split


def _teacher_reference(offset_metadata: Mapping,
                       override):
    stored_path = offset_metadata.get('pretrained_ground_ckpt_dir')
    step = offset_metadata.get('ground_checkpoint_step')
    fingerprint = offset_metadata.get('ground_checkpoint_fingerprint')
    if stored_path is None or step is None or fingerprint is None:
        raise ValueError('Delta-offset checkpoint is missing its exact frozen-teacher reference '
                         '(path, step, or fingerprint).')

    teacher_dir = Path(override if override is not None else stored_path).absolute().resolve()
    if not teacher_dir.exists():
        raise FileNotFoundError(f'Frozen ground teacher checkpoint does not exist: {teacher_dir}.')
    step = int(step)
    teacher_params = load_params_from_ckpt_dir(teacher_dir, step=step)
    actual_fingerprint = checkpoint_fingerprint(
        teacher_dir,
        step=step,
        params=teacher_params)
    if actual_fingerprint != fingerprint:
        raise ValueError('Frozen ground teacher fingerprint mismatch: the selected checkpoint is not '
                         'the teacher used to define these offset targets.')
    return teacher_dir, step, str(actual_fingerprint), teacher_params


def evaluate_delta_offset():
    parser = argparse.ArgumentParser(
        description='Evaluate a state-conditioned delta-offset SO3krates checkpoint.')
    parser.add_argument('--delta_offset_ckpt_dir', type=str, default=os.getcwd())
    parser.add_argument('--pretrained_ground_ckpt_dir', '--ground_ckpt_dir',
                        dest='pretrained_ground_ckpt_dir', type=str, default=None)
    parser.add_argument('--apply_to', type=str, default=None)
    parser.add_argument('--on', choices=('train', 'valid', 'test'), default='test')
    parser.add_argument('--n_test', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--from_split', type=str, default=None)
    parser.add_argument('--units', action=StoreDictKeyPair,
                        metavar='KEY1=VAL1,KEY2=VAL2...', default=None)
    parser.add_argument('--prop_keys', action=StoreDictKeyPair,
                        metavar='KEY1=VAL1,KEY2=VAL2...', default=None)
    parser.add_argument('--jax_dtype', choices=('x32', 'x64'), default='x32')
    parser.add_argument('--save_predictions_to', type=str,
                        default='delta_offset_predictions.npz')
    args = parser.parse_args()

    if args.jax_dtype == 'x64':
        from jax import config
        config.update('jax_enable_x64', True)

    offset_ckpt_dir = Path(args.delta_offset_ckpt_dir).absolute().resolve()
    offset_h = read_json(offset_ckpt_dir / 'hyperparameters.json')
    offset_metadata = _require_delta_offset_checkpoint(offset_h, offset_ckpt_dir)
    trained_states = tuple(int(state) for state in offset_metadata.get('state_ids', ()))

    teacher_dir, teacher_step, teacher_fingerprint, teacher_params = _teacher_reference(
        offset_metadata,
        override=args.pretrained_ground_ckpt_dir)
    teacher_h = read_json(teacher_dir / 'hyperparameters.json')
    if 'stack_net' not in teacher_h or teacher_h.get('training_mode') in ('delta', 'delta_offset'):
        raise ValueError('The pinned teacher must be an ordinary ground-state StackNet checkpoint.')
    if is_bond_aware_stacknet_metadata(teacher_h):
        raise ValueError('The pinned delta-offset teacher must be non-bond-aware; only the '
                         'trainable offset student may consume active-state bond descriptors.')

    offset_net = init_delta_offset_model(offset_h)
    teacher_net = init_stack_net(teacher_h)
    prop_keys = dict(offset_net.prop_keys)
    if args.prop_keys is not None:
        prop_keys.update(args.prop_keys)
        offset_net.reset_prop_keys(prop_keys=prop_keys)
    # Both models consume one canonical batch even if the original teacher used
    # different physical dataset aliases.
    teacher_net.reset_prop_keys(prop_keys=prop_keys)

    backbone_h = offset_metadata['backbone']
    bond_aware = is_bond_aware_stacknet_metadata({'stack_net': backbone_h})
    r_cut, mic = _geometry_settings(backbone_h)
    if bond_aware and mic:
        raise ValueError('Bond-aware delta-offset evaluation supports only nonperiodic checkpoints.')

    coach = Coach(**offset_h['coach'])
    inputs = list(coach.inputs)
    for required_input in (pn.atomic_type, pn.atomic_position, pn.active_state):
        if required_input not in inputs:
            inputs.append(required_input)

    conversion_table = {}
    if args.units is not None:
        for quantity, value in args.units.items():
            conversion_table[prop_keys[quantity]] = eval(value)  # noqa: S307 - established CLI contract

    apply_to, from_split = _resolve_data_path(args, coach)
    data, graph_metadata = _load_excited_npz_data(
        path=apply_to,
        conversion_table=conversion_table,
        inputs=inputs,
        prop_keys=prop_keys,
        trained_states=trained_states,
        bond_aware=bond_aware,
        r_cut=r_cut,
        mic=mic,
        bond_descriptor_mode=offset_metadata.get('bond_descriptor_mode', 'absolute_state'))
    data_set = DataSet(data=data, prop_keys=prop_keys, graph_metadata=graph_metadata)
    split = _evaluation_split(data_set=data_set,
                              ckpt_dir=offset_ckpt_dir,
                              evaluate_on=args.on,
                              from_split=from_split,
                              n_test=args.n_test,
                              r_cut=r_cut,
                              mic=mic,
                              bond_aware=bond_aware)

    data_tuple = DataTuple(inputs=inputs,
                           targets=(pn.energy, pn.force),
                           prop_keys=prop_keys)
    evaluation_inputs, active_targets = data_tuple(split)
    source_indices = np.asarray(split[SOURCE_INDEX_KEY]).reshape(-1)

    offset_params = load_params_from_ckpt_dir(offset_ckpt_dir)
    teacher_scales = read_json(teacher_dir / 'scales.json')
    teacher_obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(teacher_net), in_axes=(None, 0)))
    offset_obs_fn = jax.jit(
        jax.vmap(get_delta_offset_energy_force_fn(offset_net), in_axes=(None, 0)))

    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    offset_energy_key = prop_keys[pn.offset_energy]
    offset_force_key = prop_keys[pn.offset_force]

    def combined_obs_fn(params, batch_inputs):
        teacher_variables, offset_variables = params
        teacher_outputs = teacher_obs_fn(teacher_variables, batch_inputs)
        offset_outputs = offset_obs_fn(offset_variables, batch_inputs)
        teacher_energy, teacher_force = restore_ground_prediction_units(
            teacher_outputs,
            batch_inputs,
            teacher_scales,
            prop_keys)
        return {
            'teacher_energy': teacher_energy,
            'teacher_force': teacher_force,
            'offset_energy': offset_outputs[offset_energy_key],
            'offset_force': offset_outputs[offset_force_key],
        }

    predictions = _batched_predictions(
        params=(teacher_params, offset_params),
        obs_fn=combined_obs_fn,
        inputs=evaluation_inputs,
        batch_size=args.batch_size)
    arrays = _offset_evaluation_arrays(
        teacher_energy=predictions['teacher_energy'],
        teacher_force=predictions['teacher_force'],
        predicted_offset_energy=predictions['offset_energy'],
        predicted_offset_force=predictions['offset_force'],
        target_energy=active_targets[energy_key],
        target_force=active_targets[force_key])

    state_key = prop_keys[pn.active_state]
    z_key = prop_keys[pn.atomic_type]
    node_mask_key = prop_keys[pn.node_mask]
    if node_mask_key in evaluation_inputs:
        node_mask = np.asarray(evaluation_inputs[node_mask_key]).astype(bool)
    else:
        node_mask = np.asarray(evaluation_inputs[z_key]) != 0
    active_states = _state_vector(evaluation_inputs[state_key])
    metrics = _delta_offset_metrics(
        arrays=arrays,
        active_state=active_states,
        node_mask=node_mask,
        trained_states=trained_states)

    report = {
        'training_mode': 'delta_offset',
        'offset_sign_convention': OFFSET_SIGN_CONVENTION,
        'evaluated_split': args.on,
        'evaluation_data_path': Path(apply_to).absolute().resolve().as_posix(),
        'n_structures': int(len(active_states)),
        'state_counts': {
            str(state): int(np.sum(active_states == state))
            for state in trained_states
            if np.any(active_states == state)
        },
        'teacher_checkpoint': {
            'path': teacher_dir.as_posix(),
            'step': teacher_step,
            'fingerprint': teacher_fingerprint,
        },
        'metrics': metrics,
    }
    pprint(report)
    with open(offset_ckpt_dir / f'delta_offset_metrics_on_{args.on}.json', 'w') as handle:
        json.dump(report, handle, indent=1)

    if args.save_predictions_to is not None:
        position_key = prop_keys[pn.atomic_position]
        payload = {
            'source_index': source_indices,
            'active_state': active_states,
            'R': np.asarray(evaluation_inputs[position_key]),
            'z': np.asarray(evaluation_inputs[z_key]),
            'node_mask': node_mask,
            'teacher_checkpoint_path': np.asarray(teacher_dir.as_posix()),
            'teacher_checkpoint_step': np.asarray(teacher_step),
            'teacher_checkpoint_fingerprint': np.asarray(teacher_fingerprint),
            'offset_sign_convention': np.asarray(OFFSET_SIGN_CONVENTION),
            'evaluation_data_path': np.asarray(Path(apply_to).absolute().resolve().as_posix()),
            **arrays,
        }
        np.savez(offset_ckpt_dir / args.save_predictions_to, **payload)


if __name__ == '__main__':
    evaluate_delta_offset()
