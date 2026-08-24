import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from pathlib import Path
from pprint import pprint
from typing import Dict

from ase.units import *

from mlff.cAPI.mlff_eval import is_bond_aware_stacknet_metadata, unit_convert_data
from mlff.cAPI.process_argparse import StoreDictKeyPair
from mlff.data import (DataSet, DataTuple, load_precomputed_graph_metadata,
                       select_data_for_model)
from mlff.inference.evaluation import evaluate_model, mae_metric, rmse_metric, r2_metric
from mlff.io import load_params_from_ckpt_dir, read_json
from mlff.nn import init_delta_model
from mlff.nn.representation.delta import ground_teacher_inputs
from mlff.nn.stacknet import get_delta_energy_force_fn, get_obs_and_force_fn, init_stack_net
from mlff.properties import property_names as pn
from mlff.training import Coach


ENERGY_OUTPUTS = ('E0', 'E1', 'E2', 'Delta_E1', 'Delta_E2')
FORCE_OUTPUTS = ('F0', 'F1', 'F2', 'Delta_F1', 'Delta_F2')


def _geometry_settings(stack_h: Dict):
    # Resolve graph construction from serialized geometry metadata instead of evaluation-time defaults.
    geometry_h = next(embedding_h['geometry_embed']
                      for embedding_h in stack_h['geometry_embeddings']
                      if 'geometry_embed' in embedding_h)
    return float(geometry_h['r_cut']), bool(geometry_h.get('mic', False))


def _load_npz_data(path, conversion_table, inputs, targets, prop_keys, bond_aware, r_cut):
    if Path(path).suffix != '.npz':
        raise ValueError('Delta evaluation requires an NPZ containing ground and delta targets.')

    # Keep the same fixed-graph validation contract used during delta training.
    data = dict(np.load(path, allow_pickle=False))
    data = unit_convert_data(data, table=conversion_table)
    if bond_aware:
        # Fixed-graph inputs already exist at the NPZ boundary and can be selected before dataset validation.
        data = select_data_for_model(data=data,
                                     inputs=inputs,
                                     targets=targets,
                                     prop_keys=prop_keys)
    graph_metadata = (load_precomputed_graph_metadata(path, r_cut=r_cut)
                      if bond_aware else None)
    return data, graph_metadata


def _evaluation_split(data_set,
                      ckpt_dir,
                      evaluate_on,
                      from_split,
                      n_test,
                      r_cut,
                      mic,
                      bond_aware):
    if from_split is not None:
        # Replay the requested saved split so delta and ground checkpoints see identical structures.
        split_counts = {'train': (None, 0, 0),
                        'valid': (0, None, 0),
                        'test': (0, 0, n_test)}
        n_train, n_valid, resolved_n_test = split_counts[evaluate_on]
        data_set.load_split(file=os.path.join(ckpt_dir, 'splits.json'),
                            n_train=n_train,
                            n_valid=n_valid,
                            n_test=resolved_n_test,
                            r_cut=r_cut,
                            mic=mic,
                            split_name=from_split,
                            precomputed_graph=bond_aware)
        return data_set.get_data_split()[evaluate_on]

    # Treat an explicitly supplied file as an independent test set and select all rows by default.
    n_apply = data_set.n_data if n_test is None else n_test
    if n_apply < 0 or n_apply > data_set.n_data:
        raise ValueError(f'`--n_test` must select between 0 and {data_set.n_data} structures.')
    data_set.index_split(data_idx_train=[],
                         data_idx_valid=[],
                         data_idx_test=np.arange(n_apply),
                         r_cut=r_cut,
                         mic=mic,
                         training=False,
                         precomputed_graph=bond_aware)
    return data_set.get_data_split()['test']


def _restore_ground_units(outputs, inputs, scales, prop_keys):
    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    z_key = prop_keys[pn.atomic_type]

    # Reverse the target normalization recorded by ordinary ground-state training.
    energy_scale = jnp.asarray(scales[pn.energy]['scale'])
    energy_shifts = jnp.asarray(scales[pn.energy]['per_atom_shift'])
    structure_shift = jnp.take(energy_shifts, inputs[z_key].astype(jnp.int32)).sum(axis=-1)
    energy = energy_scale * outputs[energy_key] + structure_shift[:, None]

    # Ground force targets are only multiplicatively scaled by the existing dataset contract.
    force_scale = jnp.asarray(scales[pn.force]['scale'])
    force = force_scale * outputs[force_key]
    return energy, force


def _combined_targets(targets, prop_keys):
    energy_0 = targets[prop_keys[pn.energy]]
    force_0 = targets[prop_keys[pn.force]]
    delta_energy_1 = targets[prop_keys[pn.delta_energy_1]]
    delta_energy_2 = targets[prop_keys[pn.delta_energy_2]]
    delta_force_1 = targets[prop_keys[pn.delta_force_1]]
    delta_force_2 = targets[prop_keys[pn.delta_force_2]]

    # Reconstruct reference excited states from the same ground-plus-correction definition used for predictions.
    return {'E0': energy_0,
            'E1': energy_0 + delta_energy_1,
            'E2': energy_0 + delta_energy_2,
            'Delta_E1': delta_energy_1,
            'Delta_E2': delta_energy_2,
            'F0': force_0,
            'F1': force_0 + delta_force_1,
            'F2': force_0 + delta_force_2,
            'Delta_F1': delta_force_1,
            'Delta_F2': delta_force_2}


def _metric_triplet(prediction, target):
    # Reuse the repository's MAE/RMSE/R2 definitions after applying the explicit delta masks.
    with np.errstate(divide='ignore', invalid='ignore'):
        return {'mae': float(mae_metric(prediction=prediction, target=target)),
                'rmse': float(rmse_metric(prediction=prediction, target=target)),
                'R2': float(r2_metric(prediction=prediction, target=target))}


def _delta_metrics(predictions, targets, inputs, prop_keys):
    node_mask_key = prop_keys[pn.node_mask]
    if node_mask_key in inputs:
        node_mask = np.asarray(inputs[node_mask_key]).astype(bool)
    else:
        # Fall back to the package-wide zero-atomic-number padding convention.
        node_mask = np.asarray(inputs[prop_keys[pn.atomic_type]]) != 0

    metrics = {'mae': {}, 'rmse': {}, 'R2': {}}
    for name in (*ENERGY_OUTPUTS, *FORCE_OUTPUTS):
        prediction = np.asarray(predictions[name])
        target = np.asarray(targets[name])
        finite = np.isfinite(target)
        if name in FORCE_OUTPUTS:
            finite = finite & np.broadcast_to(node_mask[..., None], target.shape)

        # Flatten only valid entries so padded atoms and missing labels cannot bias aggregate metrics.
        values = _metric_triplet(prediction[finite], target[finite])
        for metric_name, value in values.items():
            metrics[metric_name][name] = value
    return metrics


def evaluate_delta():
    parser = argparse.ArgumentParser(description='Evaluate ground and delta SO3krates checkpoints.')
    parser.add_argument('--delta_ckpt_dir', type=str, default=os.getcwd())
    parser.add_argument('--ground_ckpt_dir', type=str, default=None)
    parser.add_argument('--apply_to', type=str, default=None)
    parser.add_argument('--on', choices=('train', 'valid', 'test'), default='test')
    parser.add_argument('--n_test', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--from_split', type=str, default=None)
    parser.add_argument('--units', action=StoreDictKeyPair,
                        metavar='KEY1=VAL1,KEY2=VAL2...', default=None)
    parser.add_argument('--prop_keys', action=StoreDictKeyPair,
                        metavar='KEY1=VAL1,KEY2=VAL2...', default=None)
    parser.add_argument('--jax_dtype', type=str, default='x32')
    parser.add_argument('--save_predictions_to', type=str, default='delta_predictions.npz')
    args = parser.parse_args()

    if args.jax_dtype == 'x64':
        # Match the standard evaluator's opt-in double-precision behavior.
        from jax import config
        config.update('jax_enable_x64', True)

    delta_ckpt_dir = Path(args.delta_ckpt_dir).absolute().resolve()
    delta_h = read_json(delta_ckpt_dir / 'hyperparameters.json')
    if 'delta_model' not in delta_h:
        raise ValueError(f'{delta_ckpt_dir} does not contain delta-model hyperparameters.')

    # Default to the exact ground checkpoint recorded when delta training began.
    ground_ckpt_arg = args.ground_ckpt_dir
    if ground_ckpt_arg is None:
        ground_ckpt_arg = delta_h['delta_model'].get('pretrained_ground_ckpt_dir')
    if ground_ckpt_arg is None:
        raise ValueError('Provide `--ground_ckpt_dir`; the delta checkpoint does not record one.')
    ground_ckpt_dir = Path(ground_ckpt_arg).absolute().resolve()
    ground_h = read_json(ground_ckpt_dir / 'hyperparameters.json')

    delta_net = init_delta_model(delta_h)
    ground_net = init_stack_net(ground_h)
    prop_keys = dict(delta_net.prop_keys)
    if args.prop_keys is not None:
        # Apply dataset aliases to both models so reconstruction uses one canonical batch.
        prop_keys.update(args.prop_keys)
        delta_net.reset_prop_keys(prop_keys=prop_keys)
        ground_net.reset_prop_keys(prop_keys=prop_keys)

    delta_stack_h = delta_h['delta_model']['backbone']
    bond_aware = is_bond_aware_stacknet_metadata({'stack_net': delta_stack_h})
    ground_bond_aware = is_bond_aware_stacknet_metadata(ground_h)
    r_cut, mic = _geometry_settings(delta_stack_h)
    if bond_aware and mic:
        raise ValueError('Bond-aware delta evaluation currently supports only nonperiodic checkpoints.')

    delta_coach = Coach(**delta_h['coach'])
    inputs = list(delta_coach.inputs)
    targets = [pn.energy, pn.force, pn.delta_energy_1, pn.delta_energy_2,
               pn.delta_force_1, pn.delta_force_2]

    # Resolve unit conversion factors through semantic property mappings.
    conversion_table = {}
    if args.units is not None:
        for quantity, value in args.units.items():
            conversion_table[prop_keys[quantity]] = eval(value)

    apply_to = args.apply_to
    from_split = args.from_split
    if apply_to is None and delta_coach.data_path is not None:
        # A single-source training run can replay its saved split by default.
        apply_to = delta_coach.data_path
        from_split = 'split' if from_split is None else from_split
    elif apply_to is None:
        # Separate-file training runs evaluate the matching source file without a synthetic saved split.
        apply_to = (delta_coach.train_data_path
                    if args.on == 'train' else delta_coach.valid_data_path)
    if apply_to is None:
        raise ValueError('No evaluation data path was supplied or stored in the delta checkpoint.')

    data, graph_metadata = _load_npz_data(path=apply_to,
                                          conversion_table=conversion_table,
                                          inputs=inputs,
                                          targets=targets,
                                          prop_keys=prop_keys,
                                          bond_aware=bond_aware,
                                          r_cut=r_cut)
    data_set = DataSet(data=data, prop_keys=prop_keys, graph_metadata=graph_metadata)
    split = _evaluation_split(data_set=data_set,
                              ckpt_dir=delta_ckpt_dir,
                              evaluate_on=args.on,
                              from_split=from_split,
                              n_test=args.n_test,
                              r_cut=r_cut,
                              mic=mic,
                              bond_aware=bond_aware)

    data_tuple = DataTuple(inputs=inputs, targets=targets, prop_keys=prop_keys)
    test_inputs, correction_targets = data_tuple(split)
    combined_targets = _combined_targets(correction_targets, prop_keys=prop_keys)

    delta_params = load_params_from_ckpt_dir(delta_ckpt_dir)
    ground_params = load_params_from_ckpt_dir(ground_ckpt_dir)
    ground_scales = read_json(ground_ckpt_dir / 'scales.json')
    ground_obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(ground_net), in_axes=(None, 0)))
    delta_obs_fn = jax.jit(jax.vmap(get_delta_energy_force_fn(delta_net), in_axes=(None, 0)))

    def combined_obs_fn(params, batch_inputs):
        ground_variables, delta_variables = params
        ground_outputs = ground_obs_fn(
            ground_variables,
            ground_teacher_inputs(batch_inputs, prop_keys, ground_bond_aware))
        delta_outputs = delta_obs_fn(delta_variables, batch_inputs)
        energy_0, force_0 = _restore_ground_units(ground_outputs,
                                                  batch_inputs,
                                                  ground_scales,
                                                  prop_keys)
        delta_energy_1 = delta_outputs[prop_keys[pn.delta_energy_1]]
        delta_energy_2 = delta_outputs[prop_keys[pn.delta_energy_2]]
        delta_force_1 = delta_outputs[prop_keys[pn.delta_force_1]]
        delta_force_2 = delta_outputs[prop_keys[pn.delta_force_2]]

        # Reconstruct both excited states only after ground predictions have returned to raw dataset units.
        return {'E0': energy_0,
                'E1': energy_0 + delta_energy_1,
                'E2': energy_0 + delta_energy_2,
                'Delta_E1': delta_energy_1,
                'Delta_E2': delta_energy_2,
                'F0': force_0,
                'F1': force_0 + delta_force_1,
                'F2': force_0 + delta_force_2,
                'Delta_F1': delta_force_1,
                'Delta_F2': delta_force_2}

    _, evaluation_arrays = evaluate_model(params=(ground_params, delta_params),
                                          obs_fn=combined_obs_fn,
                                          data=(test_inputs, combined_targets),
                                          batch_size=args.batch_size,
                                          metric_fn=None)
    metrics = _delta_metrics(predictions=evaluation_arrays['predictions'],
                             targets=evaluation_arrays['targets'],
                             inputs=evaluation_arrays['inputs'],
                             prop_keys=prop_keys)
    pprint(metrics)

    # Save delta metrics and flat prediction/target arrays beside the delta checkpoint.
    metrics_path = delta_ckpt_dir / f'delta_metrics_on_{args.on}.json'
    with open(metrics_path, 'w') as handle:
        json.dump(metrics, handle, indent=1)
    if args.save_predictions_to is not None:
        save_payload = {}
        for name in (*ENERGY_OUTPUTS, *FORCE_OUTPUTS):
            save_payload[f'prediction_{name}'] = evaluation_arrays['predictions'][name]
            save_payload[f'target_{name}'] = evaluation_arrays['targets'][name]
        np.savez(delta_ckpt_dir / args.save_predictions_to, **save_payload)


if __name__ == '__main__':
    evaluate_delta()
