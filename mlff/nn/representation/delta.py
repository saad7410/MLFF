import copy
import logging

import jax.numpy as jnp
import flax.linen as nn

from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Any, Dict, Tuple

from mlff.masking.mask import safe_scale
from mlff.nn.observable import StateDeltaHead
from mlff.nn.stacknet.stacknet import StackNet, init_stack_net
from mlff.properties import property_names as pn


ABSOLUTE_BOND_DESCRIPTOR = 'absolute_state'
RELATIVE_BOND_DESCRIPTOR = 'relative_to_s0'
LEGACY_BOND_LAYOUT = 'legacy_auto'
NAMED_BOND_LAYOUT = 'named_v1'
RELATIVE_BOND_FEATURE_DIM = 12


def build_relative_bond_descriptor(bond_prob_0,
                                   bond_mask_0,
                                   bond_prob_state,
                                   bond_mask_state):
    """Build [b0, bs, bs-b0] after clearing unannotated state-specific edges."""
    if bond_prob_0.shape != bond_prob_state.shape or bond_prob_0.shape[-1] != 4:
        raise ValueError('State-0 and excited-state bond probabilities must share shape (..., P, 4).')
    if bond_mask_0.shape != bond_prob_0.shape[:-1] or bond_mask_state.shape != bond_prob_state.shape[:-1]:
        raise ValueError('State bond masks must match their probability edge dimensions.')

    masked_prob_0 = safe_scale(bond_prob_0, scale=bond_mask_0[..., None])
    masked_prob_state = safe_scale(bond_prob_state, scale=bond_mask_state[..., None])
    descriptor = jnp.concatenate((masked_prob_0,
                                  masked_prob_state,
                                  masked_prob_state - masked_prob_0),
                                 axis=-1)
    descriptor_mask = jnp.logical_or(bond_mask_0.astype(bool), bond_mask_state.astype(bool))
    return descriptor, descriptor_mask


def upgrade_stacknet_for_relative_bond_delta(ground_h: Dict) -> Dict:
    """Clone ordinary StackNet metadata and add versioned 12-channel delta-only bond branches."""
    upgraded_h = copy.deepcopy(ground_h)
    if 'stack_net' not in upgraded_h:
        raise KeyError('Ground checkpoint metadata is missing `stack_net`.')

    layers = upgraded_h['stack_net'].get('layers', ())
    if not layers:
        raise ValueError('Ground checkpoint does not contain SO3krates representation layers.')

    for layer_index, layer_h in enumerate(layers):
        if 'so3krates_layer' not in layer_h:
            raise ValueError(f'Delta bond upgrade requires SO3krates layers; layer {layer_index} is incompatible.')
        so3_h = layer_h['so3krates_layer']
        if so3_h.get('bond_aware', False):
            raise ValueError('Relative bond upgrade is only defined for an ordinary non-bond-aware ground backbone.')
        if so3_h.get('fb_filter', 'radial_spherical') != 'radial_spherical':
            raise ValueError('Relative bond upgrade requires radial-spherical feature filters.')
        if so3_h.get('gb_filter', 'radial_spherical') != 'radial_spherical':
            raise ValueError('Relative bond upgrade requires radial-spherical geometric filters.')
        so3_h['bond_aware'] = True
        so3_h['bond_feature_dim'] = RELATIVE_BOND_FEATURE_DIM
        so3_h['bond_parameter_layout'] = NAMED_BOND_LAYOUT
    return upgraded_h


class StateSpecificDeltaSo3krates(nn.Module):
    backbone: StackNet
    delta_head: StateDeltaHead
    prop_keys: Dict
    pretrained_ground_ckpt_dir: str = None
    freeze_pretrained_backbone: bool = False
    bond_backbone_upgrade: bool = False
    bond_descriptor_mode: str = ABSOLUTE_BOND_DESCRIPTOR
    bond_feature_dim: int = 4
    bond_parameter_layout: str = LEGACY_BOND_LAYOUT

    def setup(self):
        required_outputs = (pn.delta_energy_1, pn.delta_energy_2,
                            pn.delta_force_1, pn.delta_force_2)
        missing_outputs = [name for name in required_outputs if name not in self.prop_keys]
        if missing_outputs:
            raise KeyError(f'Delta SO3krates is missing property mappings for {missing_outputs}.')
        if self.bond_descriptor_mode not in (ABSOLUTE_BOND_DESCRIPTOR, RELATIVE_BOND_DESCRIPTOR):
            raise ValueError(f'Unknown delta bond descriptor mode `{self.bond_descriptor_mode}`.')

        bond_layers = [layer for layer in self.backbone.layers if getattr(layer, 'bond_aware', False)]
        if self.bond_feature_dim <= 0:
            raise ValueError('`bond_feature_dim` must be positive.')
        if self.bond_parameter_layout not in (LEGACY_BOND_LAYOUT, NAMED_BOND_LAYOUT):
            raise ValueError(f'Unknown delta bond parameter layout `{self.bond_parameter_layout}`.')
        incompatible_bond_layers = [
            (getattr(layer, 'bond_feature_dim', 4),
             getattr(layer, 'bond_parameter_layout', LEGACY_BOND_LAYOUT))
            for layer in bond_layers
            if (getattr(layer, 'bond_feature_dim', 4) != self.bond_feature_dim
                or getattr(layer, 'bond_parameter_layout', LEGACY_BOND_LAYOUT)
                != self.bond_parameter_layout)
        ]
        if incompatible_bond_layers:
            raise ValueError('Delta bond metadata must match every bond-aware backbone layer.')
        if self.bond_descriptor_mode == RELATIVE_BOND_DESCRIPTOR:
            if len(bond_layers) != len(self.backbone.layers):
                raise ValueError('Relative delta descriptors require every backbone layer to be bond-aware.')
            invalid_dims = [getattr(layer, 'bond_feature_dim', 4)
                            for layer in bond_layers
                            if getattr(layer, 'bond_feature_dim', 4) != RELATIVE_BOND_FEATURE_DIM]
            if invalid_dims:
                raise ValueError('Relative delta descriptors require 12-channel bond-aware layers.')

    def _state_inputs(self, inputs: Dict, state: int) -> Dict:
        # Non-bond-aware checkpoints share geometry inputs and specialize only through the state embedding.
        bond_aware = any(getattr(layer, 'bond_aware', False) for layer in self.backbone.layers)
        if not bond_aware:
            return inputs

        bond_prob_name = getattr(pn, f'bond_prob_s{state}')
        bond_mask_name = getattr(pn, f'bond_mask_s{state}')
        bond_prob_key = self.prop_keys[bond_prob_name]
        bond_mask_key = self.prop_keys[bond_mask_name]
        missing_keys = [key for key in (bond_prob_key, bond_mask_key) if key not in inputs]
        if missing_keys:
            raise KeyError(f'Delta state {state} requires edge descriptors {missing_keys}.')

        # Remap state-specific chemistry onto the canonical bond keys consumed by the shared backbone.
        state_inputs = dict(inputs)
        if self.bond_descriptor_mode == RELATIVE_BOND_DESCRIPTOR:
            bond_prob_0_key = self.prop_keys[pn.bond_prob_s0]
            bond_mask_0_key = self.prop_keys[pn.bond_mask_s0]
            missing_ground_keys = [key for key in (bond_prob_0_key, bond_mask_0_key) if key not in inputs]
            if missing_ground_keys:
                raise KeyError(f'Relative delta state {state} requires state-0 descriptors {missing_ground_keys}.')
            descriptor, descriptor_mask = build_relative_bond_descriptor(
                bond_prob_0=inputs[bond_prob_0_key],
                bond_mask_0=inputs[bond_mask_0_key],
                bond_prob_state=inputs[bond_prob_key],
                bond_mask_state=inputs[bond_mask_key])
            state_inputs[self.prop_keys[pn.bond_prob]] = descriptor
            state_inputs[self.prop_keys[pn.bond_mask]] = descriptor_mask
        else:
            state_inputs[self.prop_keys[pn.bond_prob]] = inputs[bond_prob_key]
            state_inputs[self.prop_keys[pn.bond_mask]] = inputs[bond_mask_key]
        return state_inputs

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        # Run the same representation parameters with state-1 and state-2 bond descriptors.
        state_1_quantities = self.backbone.forward_features(self._state_inputs(inputs, state=1))
        state_2_quantities = self.backbone.forward_features(self._state_inputs(inputs, state=2))

        # Apply one shared state-conditioned head to both backbone passes.
        delta_e_1 = self.delta_head(state_1_quantities, state=1)
        delta_e_2 = self.delta_head(state_2_quantities, state=2)
        return {self.prop_keys[pn.delta_energy_1]: delta_e_1,
                self.prop_keys[pn.delta_energy_2]: delta_e_2}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        # Persist the exact ground representation metadata so delta checkpoints reconstruct independently.
        backbone_h = self.backbone.__dict_repr__()['stack_net']
        delta_head_h = self.delta_head.__dict_repr__()[self.delta_head.module_name]
        return {'delta_model': {'backbone': backbone_h,
                                'delta_head': delta_head_h,
                                'prop_keys': self.prop_keys,
                                'pretrained_ground_ckpt_dir': self.pretrained_ground_ckpt_dir,
                                'freeze_pretrained_backbone': self.freeze_pretrained_backbone,
                                'bond_backbone_upgrade': self.bond_backbone_upgrade,
                                'bond_descriptor_mode': self.bond_descriptor_mode,
                                'bond_feature_dim': self.bond_feature_dim,
                                'bond_parameter_layout': self.bond_parameter_layout}}

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        # Keep dataset key overrides synchronized across the wrapper, backbone, and delta head.
        self.prop_keys.update(prop_keys)
        if sub_modules:
            self.backbone.reset_prop_keys(prop_keys=prop_keys)
            self.delta_head.reset_prop_keys(prop_keys=prop_keys)


def init_state_specific_delta_so3krates(backbone: StackNet,
                                        pretrained_ground_ckpt_dir: str = None,
                                        freeze_pretrained_backbone: bool = False,
                                        bond_backbone_upgrade: bool = False,
                                        bond_descriptor_mode: str = ABSOLUTE_BOND_DESCRIPTOR,
                                        bond_feature_dim: int = None,
                                        bond_parameter_layout: str = None,
                                        n_states: int = 3) -> StateSpecificDeltaSo3krates:
    feature_dim = getattr(backbone.feature_embeddings[0], 'features', None)
    if feature_dim is None:
        raise ValueError('The delta head requires a backbone feature embedding with a `features` attribute.')

    # Match the delta readout width to the serialized ground-state representation width.
    prop_keys = dict(backbone.prop_keys)
    bond_layers = [layer for layer in backbone.layers if getattr(layer, 'bond_aware', False)]
    if bond_feature_dim is None:
        feature_dims = {getattr(layer, 'bond_feature_dim', 4) for layer in bond_layers}
        if len(feature_dims) > 1:
            raise ValueError('Every delta bond-aware layer must use the same descriptor width.')
        bond_feature_dim = next(iter(feature_dims), 4)
    if bond_parameter_layout is None:
        parameter_layouts = {
            getattr(layer, 'bond_parameter_layout', LEGACY_BOND_LAYOUT)
            for layer in bond_layers
        }
        if len(parameter_layouts) > 1:
            raise ValueError('Every delta bond-aware layer must use the same parameter layout.')
        bond_parameter_layout = next(iter(parameter_layouts), LEGACY_BOND_LAYOUT)
    delta_head = StateDeltaHead(feature_dim=int(feature_dim),
                                prop_keys=prop_keys,
                                n_states=n_states)
    return StateSpecificDeltaSo3krates(backbone=backbone,
                                       delta_head=delta_head,
                                       prop_keys=prop_keys,
                                       pretrained_ground_ckpt_dir=pretrained_ground_ckpt_dir,
                                       freeze_pretrained_backbone=freeze_pretrained_backbone,
                                       bond_backbone_upgrade=bond_backbone_upgrade,
                                       bond_descriptor_mode=bond_descriptor_mode,
                                       bond_feature_dim=bond_feature_dim,
                                       bond_parameter_layout=bond_parameter_layout)


def init_delta_model(h: Dict,
                     freeze_pretrained_backbone: bool = None) -> StateSpecificDeltaSo3krates:
    delta_h = h['delta_model']

    # Reconstruct the shared representation and state head from delta checkpoint metadata.
    backbone = init_stack_net({'stack_net': delta_h['backbone']})
    delta_head = StateDeltaHead(**delta_h['delta_head'])
    freeze_backbone = (delta_h.get('freeze_pretrained_backbone', False)
                       if freeze_pretrained_backbone is None else freeze_pretrained_backbone)
    return StateSpecificDeltaSo3krates(
        backbone=backbone,
        delta_head=delta_head,
        prop_keys=delta_h['prop_keys'],
        pretrained_ground_ckpt_dir=delta_h.get('pretrained_ground_ckpt_dir'),
        freeze_pretrained_backbone=freeze_backbone,
        bond_backbone_upgrade=delta_h.get('bond_backbone_upgrade', False),
        bond_descriptor_mode=delta_h.get('bond_descriptor_mode', ABSOLUTE_BOND_DESCRIPTOR),
        bond_feature_dim=delta_h.get('bond_feature_dim', 4),
        bond_parameter_layout=delta_h.get('bond_parameter_layout', LEGACY_BOND_LAYOUT))


def _partition_pretrained_backbone(delta_variables,
                                   ground_variables,
                                   allow_bond_upgrade: bool = False):
    delta_tree = unfreeze(freeze(delta_variables))
    ground_tree = unfreeze(freeze(ground_variables))
    if 'params' not in delta_tree or 'params' not in ground_tree:
        raise KeyError('Both initialized delta variables and ground checkpoint variables need a `params` collection.')
    if 'backbone' not in delta_tree['params']:
        raise KeyError('Initialized delta variables do not contain the expected `params/backbone` subtree.')

    current_flat = flatten_dict(delta_tree['params']['backbone'])
    ground_flat = flatten_dict(ground_tree['params'])
    matched = {}
    new_bond = {}
    incompatible = []
    for path, current_value in current_flat.items():
        ground_value = ground_flat.get(path)
        if ground_value is not None and jnp.shape(ground_value) == jnp.shape(current_value):
            matched[path] = ground_value
        elif allow_bond_upgrade and 'bond_rad_filter_fn' in path and ground_value is None:
            new_bond[path] = current_value
        else:
            incompatible.append(path)
    return delta_tree, current_flat, matched, new_bond, incompatible


def get_pretrained_backbone_paths(delta_variables,
                                  ground_variables,
                                  allow_bond_upgrade: bool = False) -> Tuple[Tuple[str, ...], ...]:
    """Return exact delta-backbone leaves that are shape-compatible with the ground checkpoint."""
    _, _, matched, _, incompatible = _partition_pretrained_backbone(
        delta_variables=delta_variables,
        ground_variables=ground_variables,
        allow_bond_upgrade=allow_bond_upgrade)
    if incompatible:
        preview = ', '.join('/'.join(path) for path in incompatible[:8])
        raise ValueError(f'Ground checkpoint is incompatible with delta backbone parameters: {preview}.')
    return tuple(matched.keys())


def _zero_final_bond_layers(current_flat, new_bond):
    branch_to_final_layer = {}
    for path in new_bond:
        branch_index = path.index('bond_rad_filter_fn')
        branch_path = path[:branch_index + 1]
        dense_layers = [part for part in path[branch_index + 1:]
                        if part.startswith('layers_') and part[len('layers_'):].isdigit()]
        if not dense_layers:
            raise ValueError(f'New bond parameter has no Dense layer path: {"/".join(path)}.')
        dense_index = max(int(part[len('layers_'):]) for part in dense_layers)
        branch_to_final_layer[branch_path] = max(branch_to_final_layer.get(branch_path, -1), dense_index)

    zeroed = 0
    for path in new_bond:
        branch_index = path.index('bond_rad_filter_fn')
        branch_path = path[:branch_index + 1]
        final_layer_name = f'layers_{branch_to_final_layer[branch_path]}'
        if final_layer_name in path[branch_index + 1:] and path[-1] in ('kernel', 'bias'):
            current_flat[path] = jnp.zeros_like(current_flat[path])
            zeroed += 1
    if zeroed == 0:
        raise ValueError('No final bond-branch kernels or biases were found for zero initialization.')
    return zeroed


def load_pretrained_backbone(delta_variables,
                             ground_variables,
                             strict: bool = True,
                             allow_bond_upgrade: bool = False,
                             return_transferred_paths: bool = False):
    # Convert both variable trees to mutable dictionaries for a shape-checked representation transfer.
    delta_tree, current_flat, matched, new_bond, incompatible = _partition_pretrained_backbone(
        delta_variables=delta_variables,
        ground_variables=ground_variables,
        allow_bond_upgrade=allow_bond_upgrade)

    if strict and incompatible:
        preview = ', '.join('/'.join(path) for path in incompatible[:8])
        raise ValueError(f'Ground checkpoint is incompatible with delta backbone parameters: {preview}.')
    if not matched:
        raise ValueError('Ground checkpoint did not contain any shape-compatible delta backbone parameters.')
    if allow_bond_upgrade:
        if not new_bond:
            raise ValueError('Relative bond upgrade did not initialize any new `bond_rad_filter_fn` parameters.')
        zeroed = _zero_final_bond_layers(current_flat=current_flat, new_bond=new_bond)
        logging.info('Zero-initialized %d final bond-branch parameter arrays.', zeroed)

    # Replace only compatible backbone leaves and preserve the newly initialized delta-head parameters.
    current_flat.update(matched)
    delta_tree['params']['backbone'] = unflatten_dict(current_flat)
    logging.info('Loaded %d pretrained SO3krates backbone parameter arrays.', len(matched))
    result = freeze(delta_tree)
    if return_transferred_paths:
        return result, tuple(matched.keys())
    return result
