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

from mlff.io import (create_directory, bundle_dicts, save_dict, read_json,
                     load_params_from_ckpt_dir)
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.data import DataTuple, DataSet, load_precomputed_graph_metadata, select_data_for_model
from mlff.cAPI.process_argparse import StoreDictKeyPair
from mlff.nn.stacknet import (get_obs_and_force_fn, get_observable_fn,
                              get_energy_force_stress_fn, get_delta_energy_force_fn,
                              init_stack_net)
from mlff.nn import (So3krates, init_state_specific_delta_so3krates,
                     get_pretrained_backbone_paths, init_delta_model,
                     load_pretrained_backbone, upgrade_stacknet_for_relative_bond_delta)
from mlff.nn.representation.delta import (ABSOLUTE_BOND_DESCRIPTOR,
                                          LEGACY_BOND_LAYOUT,
                                          NAMED_BOND_LAYOUT,
                                          RELATIVE_BOND_DESCRIPTOR,
                                          RELATIVE_BOND_FEATURE_DIM)
from mlff.nn.observable import Energy
from mlff.data import AseDataLoader
from mlff.properties import md17_property_keys

import mlff.properties.property_names as pn

# logging.basicConfig(level=logging.INFO)
port = portpicker.pick_unused_port()
jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)


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


def train_so3krates():
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

    parser.add_argument('--delta', action='store_true', required=False,
                        help='Train state-1/state-2 energy and force corrections from a ground SO3krates checkpoint.')
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
                            pn.delta_force_2: 0.495}
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

    # Copy parser defaults before adding opt-in delta and bond properties to avoid mutating the shared mapping.
    prop_keys = dict(args.prop_keys)
    for property_name, default_key in md17_property_keys.items():
        prop_keys.setdefault(property_name, default_key)

    delta_learning = args.delta
    restart_h = None
    if delta_learning and args.restart_from_ckpt_dir is not None:
        restart_h = read_json(Path(args.restart_from_ckpt_dir).absolute().resolve() / 'hyperparameters.json')
        if 'delta_model' not in restart_h:
            raise ValueError('A delta restart checkpoint must contain `delta_model` hyperparameters.')

    pretrained_ground_ckpt_dir = args.pretrained_ground_ckpt_dir
    if delta_learning and pretrained_ground_ckpt_dir is None and restart_h is not None:
        # Resume delta runs from their saved ground-checkpoint reference when no override was supplied.
        pretrained_ground_ckpt_dir = restart_h.get('delta_model', {}).get('pretrained_ground_ckpt_dir')
    if delta_learning and pretrained_ground_ckpt_dir is None:
        raise ValueError('`--delta` requires `--pretrained_ground_ckpt_dir` for backbone initialization.')

    ground_h = None
    if delta_learning:
        # Reconstruct architecture from the ground checkpoint instead of duplicating its dimensions in CLI flags.
        pretrained_ground_ckpt_dir = Path(pretrained_ground_ckpt_dir).absolute().resolve().as_posix()
        ground_h = read_json(Path(pretrained_ground_ckpt_dir) / 'hyperparameters.json')
        ground_bond_aware = is_bond_aware_stacknet_metadata(ground_h)
        if restart_h is not None:
            # A restart always reconstructs its saved delta architecture. CLI flags may not mutate it in place.
            saved_delta_backbone_h = {'stack_net': restart_h['delta_model']['backbone']}
            bond_aware = is_bond_aware_stacknet_metadata(saved_delta_backbone_h)
            if args.bond_aware and not bond_aware:
                raise ValueError('`--bond_aware` cannot change the architecture of a non-bond-aware delta restart.')
            bond_backbone_upgrade = restart_h['delta_model'].get('bond_backbone_upgrade', False)
            bond_descriptor_mode = restart_h['delta_model'].get('bond_descriptor_mode',
                                                                  ABSOLUTE_BOND_DESCRIPTOR)
            bond_feature_dim = restart_h['delta_model'].get('bond_feature_dim', 4)
            bond_parameter_layout = restart_h['delta_model'].get('bond_parameter_layout',
                                                                  LEGACY_BOND_LAYOUT)
        else:
            # A non-bond ground checkpoint can be upgraded only inside the new delta model.
            bond_backbone_upgrade = bool(args.bond_aware and not ground_bond_aware)
            bond_aware = bool(ground_bond_aware or args.bond_aware)
            bond_descriptor_mode = (RELATIVE_BOND_DESCRIPTOR
                                    if bond_backbone_upgrade else ABSOLUTE_BOND_DESCRIPTOR)
            bond_feature_dim = RELATIVE_BOND_FEATURE_DIM if bond_backbone_upgrade else 4
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

    if bond_aware:
        # The first bond-aware port consumes only nonperiodic NPZ files with a fixed graph.
        non_npz_files = [path for path in data_files if Path(path).suffix != '.npz']
        if non_npz_files:
            raise ValueError('`--bond_aware` accepts only NPZ input files with precomputed graph arrays.')
        if args.mic:
            raise ValueError('`--bond_aware` currently supports only nonperiodic precomputed graphs.')
        if args.restart_from_ckpt_dir is not None and not delta_learning:
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

    jax_dtype = args.jax_dtype
    if jax_dtype == 'x64':
        from jax import config
        config.update("jax_enable_x64", True)

    if delta_learning:
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
        raise ValueError('Bond-aware delta learning currently supports only nonperiodic pretrained checkpoints.')

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

    if mic:
        inputs += [pn.unit_cell]
        inputs += [pn.cell_offset]

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
        if extension == '.npz':
            # Disable object loading so the graph contract remains numeric and inspectable.
            data = dict(np.load(d, allow_pickle=False))
            if bond_aware:
                graph_metadata.append(load_precomputed_graph_metadata(d, r_cut=r_cut))
        else:
            load_stress = pn.stress in targets
            data_loader = AseDataLoader(d, load_stress=load_stress, neighbors_format='dense')
            data = data_loader.load_all()

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

    if not delta_learning:
        # Ground-state training retains its existing energy-shift behavior.
        if shift_by == 'mean':
            data_set.shift_x_by_mean_x(x=pn.energy)
        elif shift_by == 'atomic_number':
            data_set.shift_x_by_type(x=pn.energy, shifts=shifts)
        elif shift_by == 'lse':
            data_set.shift_x_by_type(x=pn.energy)

    d = data_set.get_data_split()

    scales = {}
    if args.loss_variance_scaling:
        for t in targets:
            if t == pn.stress:
                scales[prop_keys[t]] = 1 / np.nanvar(d['train'][prop_keys[t]], axis=0)
            elif t in (pn.energy, pn.delta_energy_1, pn.delta_energy_2):
                scales[prop_keys[t]] = 1 / np.nanvar(d['train'][prop_keys[t]])
            elif t in (pn.force, pn.delta_force_1, pn.delta_force_2):
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

    if delta_learning and args.zbl_repulsion:
        raise ValueError('ZBL repulsion belongs to the ground energy head and is not a delta-head option.')
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
    else:
        obs = [Energy(prop_keys=prop_keys, zbl_repulsion=args.zbl_repulsion)]
        net = So3krates(prop_keys=prop_keys,
                        F=F,
                        n_layer=L,
                        obs=obs,
                        geometry_embed_kwargs=geometry_embed_kwargs,
                        so3krates_layer_kwargs=so3krates_layer_kwargs)

    if not delta_learning and pn.force in targets:
        if pn.stress in targets:
            obs_fn = get_energy_force_stress_fn(net)
        else:
            obs_fn = get_obs_and_force_fn(net)
    elif not delta_learning:
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
        if delta_learning:
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
        if delta_learning and args.freeze_pretrained_backbone:
            ground_params = load_params_from_ckpt_dir(pretrained_ground_ckpt_dir)
            transferred_backbone_paths = get_pretrained_backbone_paths(
                delta_variables=params,
                ground_variables=ground_params,
                allow_bond_upgrade=bond_backbone_upgrade)

    frozen_param_paths = None
    if delta_learning and args.freeze_pretrained_backbone:
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
    h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])

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
