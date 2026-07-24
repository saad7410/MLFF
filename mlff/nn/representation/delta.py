import copy
import logging

import jax
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
ABSOLUTE_BOND_FEATURE_DIM = 4
DELTA_OFFSET_TRAINING_MODE = 'delta_offset'
OFFSET_SIGN_CONVENTION = 'active_minus_teacher'


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


def upgrade_stacknet_for_bond_delta(ground_h: Dict,
                                    bond_feature_dim: int,
                                    bond_parameter_layout: str = NAMED_BOND_LAYOUT) -> Dict:
    """Clone an ordinary StackNet and add new invariant bond branches."""
    if bond_feature_dim <= 0:
        raise ValueError('`bond_feature_dim` must be positive.')
    if bond_parameter_layout not in (LEGACY_BOND_LAYOUT, NAMED_BOND_LAYOUT):
        raise ValueError(f'Unknown bond parameter layout `{bond_parameter_layout}`.')

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
            raise ValueError('Bond upgrade is only defined for an ordinary non-bond-aware ground backbone.')
        if so3_h.get('fb_filter', 'radial_spherical') != 'radial_spherical':
            raise ValueError('Bond upgrade requires radial-spherical feature filters.')
        if so3_h.get('gb_filter', 'radial_spherical') != 'radial_spherical':
            raise ValueError('Bond upgrade requires radial-spherical geometric filters.')
        so3_h['bond_aware'] = True
        so3_h['bond_feature_dim'] = bond_feature_dim
        so3_h['bond_parameter_layout'] = bond_parameter_layout
    return upgraded_h


def upgrade_stacknet_for_relative_bond_delta(ground_h: Dict) -> Dict:
    """Clone ordinary StackNet metadata and add versioned 12-channel delta-only bond branches."""
    return upgrade_stacknet_for_bond_delta(
        ground_h=ground_h,
        bond_feature_dim=RELATIVE_BOND_FEATURE_DIM,
        bond_parameter_layout=NAMED_BOND_LAYOUT)


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


class StateRoutedOffsetSo3krates(nn.Module):
    """One-output offset model routed by the zero-based active electronic state."""
    backbone: StackNet
    offset_head: StateDeltaHead
    prop_keys: Dict
    pretrained_ground_ckpt_dir: str = None
    ground_checkpoint_step: int = None
    ground_checkpoint_fingerprint: str = None
    freeze_pretrained_backbone: bool = False
    bond_backbone_upgrade: bool = False
    bond_descriptor_mode: str = ABSOLUTE_BOND_DESCRIPTOR
    bond_feature_dim: int = ABSOLUTE_BOND_FEATURE_DIM
    bond_parameter_layout: str = LEGACY_BOND_LAYOUT
    state_ids: Tuple[int, ...] = (1, 2)
    offset_sign_convention: str = OFFSET_SIGN_CONVENTION
    required_dataset_keys: Tuple[str, ...] = ()
    teacher_prediction_mode: str = 'online_before_optimization'

    def setup(self):
        required = (pn.offset_energy, pn.offset_force, pn.active_state)
        missing = [name for name in required if name not in self.prop_keys]
        if missing:
            raise KeyError(f'Delta-offset SO3krates is missing property mappings for {missing}.')
        if tuple(sorted(set(self.state_ids))) != tuple(self.state_ids):
            raise ValueError('`state_ids` must be sorted and unique.')
        if not self.state_ids or any(state not in (1, 2) for state in self.state_ids):
            raise ValueError('Delta-offset learning supports non-empty state IDs drawn from (1, 2).')
        if self.offset_sign_convention != OFFSET_SIGN_CONVENTION:
            raise ValueError(f'Unsupported offset sign convention `{self.offset_sign_convention}`.')
        if self.bond_descriptor_mode not in (ABSOLUTE_BOND_DESCRIPTOR, RELATIVE_BOND_DESCRIPTOR):
            raise ValueError(f'Unknown delta-offset bond descriptor mode `{self.bond_descriptor_mode}`.')

        bond_layers = [layer for layer in self.backbone.layers if getattr(layer, 'bond_aware', False)]
        incompatible = [
            layer for layer in bond_layers
            if (getattr(layer, 'bond_feature_dim', ABSOLUTE_BOND_FEATURE_DIM) != self.bond_feature_dim
                or getattr(layer, 'bond_parameter_layout', LEGACY_BOND_LAYOUT)
                != self.bond_parameter_layout)
        ]
        if incompatible:
            raise ValueError('Delta-offset bond metadata must match every bond-aware backbone layer.')
        if self.bond_descriptor_mode == RELATIVE_BOND_DESCRIPTOR:
            if len(bond_layers) != len(self.backbone.layers):
                raise ValueError('Relative offset descriptors require every backbone layer to be bond-aware.')
            if self.bond_feature_dim != RELATIVE_BOND_FEATURE_DIM:
                raise ValueError('Relative offset descriptors require 12-channel bond-aware layers.')

    def _state_specific_value(self, inputs: Dict, property_base: str, state):
        """Select the active state's descriptor without evaluating two backbones."""
        if len(self.state_ids) == 1:
            property_name = getattr(pn, f'{property_base}_s{self.state_ids[0]}')
            key = self.prop_keys[property_name]
            if key not in inputs:
                raise KeyError(f'Delta-offset state {self.state_ids[0]} requires `{key}`.')
            return inputs[key]

        values = []
        for state_id in self.state_ids:
            property_name = getattr(pn, f'{property_base}_s{state_id}')
            key = self.prop_keys[property_name]
            if key not in inputs:
                raise KeyError(f'Delta-offset state {state_id} requires `{key}`.')
            values.append(inputs[key])
        # Only states 1 and 2 are supported, so a scalar selector is sufficient
        # and remains JIT/vmap safe.
        return jnp.where(state == self.state_ids[1], values[1], values[0])

    def _state_inputs(self, inputs: Dict, state) -> Dict:
        bond_aware = any(getattr(layer, 'bond_aware', False) for layer in self.backbone.layers)
        if not bond_aware or self.bond_descriptor_mode == ABSOLUTE_BOND_DESCRIPTOR:
            # Absolute mode consumes the active per-row canonical bond_prob/bond_mask
            # arrays directly. It does not require dense s1/s2 descriptor tensors.
            return inputs

        bond_prob_0_key = self.prop_keys[pn.bond_prob_s0]
        bond_mask_0_key = self.prop_keys[pn.bond_mask_s0]
        missing_ground = [key for key in (bond_prob_0_key, bond_mask_0_key) if key not in inputs]
        if missing_ground:
            raise KeyError(f'Relative delta-offset learning requires state-0 descriptors {missing_ground}.')

        state_prob = self._state_specific_value(inputs, 'bond_prob', state)
        state_mask = self._state_specific_value(inputs, 'bond_mask', state)
        descriptor, descriptor_mask = build_relative_bond_descriptor(
            bond_prob_0=inputs[bond_prob_0_key],
            bond_mask_0=inputs[bond_mask_0_key],
            bond_prob_state=state_prob,
            bond_mask_state=state_mask)
        state_inputs = dict(inputs)
        state_inputs[self.prop_keys[pn.bond_prob]] = descriptor
        state_inputs[self.prop_keys[pn.bond_mask]] = descriptor_mask
        return state_inputs

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        state_key = self.prop_keys[pn.active_state]
        if state_key not in inputs:
            raise KeyError(f'Delta-offset input is missing active-state array `{state_key}`.')
        state = jnp.asarray(inputs[state_key], dtype=jnp.int32).reshape(())
        quantities = self.backbone.forward_features(self._state_inputs(inputs, state=state))
        offset_energy = self.offset_head(quantities, state=state)
        return {self.prop_keys[pn.offset_energy]: offset_energy}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {'delta_offset_model': {
            'training_mode': DELTA_OFFSET_TRAINING_MODE,
            'backbone': self.backbone.__dict_repr__()['stack_net'],
            'offset_head': self.offset_head.__dict_repr__()[self.offset_head.module_name],
            'prop_keys': self.prop_keys,
            'pretrained_ground_ckpt_dir': self.pretrained_ground_ckpt_dir,
            'ground_checkpoint_step': self.ground_checkpoint_step,
            'ground_checkpoint_fingerprint': self.ground_checkpoint_fingerprint,
            'freeze_pretrained_backbone': self.freeze_pretrained_backbone,
            'bond_backbone_upgrade': self.bond_backbone_upgrade,
            'bond_descriptor_mode': self.bond_descriptor_mode,
            'bond_feature_dim': self.bond_feature_dim,
            'bond_parameter_layout': self.bond_parameter_layout,
            'state_ids': list(self.state_ids),
            'offset_sign_convention': self.offset_sign_convention,
            'normalization': {'target_shift': 'none', 'target_scale': 1.0},
            'required_dataset_keys': list(self.required_dataset_keys),
            'teacher_prediction_mode': self.teacher_prediction_mode,
        }}

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        self.prop_keys.update(prop_keys)
        if sub_modules:
            self.backbone.reset_prop_keys(prop_keys=prop_keys)
            self.offset_head.reset_prop_keys(prop_keys=prop_keys)


def init_state_routed_offset_so3krates(
        backbone: StackNet,
        pretrained_ground_ckpt_dir: str,
        ground_checkpoint_step: int,
        ground_checkpoint_fingerprint: str,
        freeze_pretrained_backbone: bool = False,
        bond_backbone_upgrade: bool = False,
        bond_descriptor_mode: str = ABSOLUTE_BOND_DESCRIPTOR,
        bond_feature_dim: int = None,
        bond_parameter_layout: str = None,
        state_ids: Tuple[int, ...] = (1, 2),
        required_dataset_keys: Tuple[str, ...] = (),
        n_states: int = 3) -> StateRoutedOffsetSo3krates:
    feature_dim = getattr(backbone.feature_embeddings[0], 'features', None)
    if feature_dim is None:
        raise ValueError('The delta-offset head requires a backbone feature embedding with `features`.')

    prop_keys = dict(backbone.prop_keys)
    bond_layers = [layer for layer in backbone.layers if getattr(layer, 'bond_aware', False)]
    if bond_feature_dim is None:
        feature_dims = {getattr(layer, 'bond_feature_dim', ABSOLUTE_BOND_FEATURE_DIM)
                        for layer in bond_layers}
        if len(feature_dims) > 1:
            raise ValueError('Every delta-offset bond-aware layer must use the same descriptor width.')
        bond_feature_dim = next(iter(feature_dims), ABSOLUTE_BOND_FEATURE_DIM)
    if bond_parameter_layout is None:
        layouts = {getattr(layer, 'bond_parameter_layout', LEGACY_BOND_LAYOUT)
                   for layer in bond_layers}
        if len(layouts) > 1:
            raise ValueError('Every delta-offset bond-aware layer must use the same parameter layout.')
        bond_parameter_layout = next(iter(layouts), LEGACY_BOND_LAYOUT)

    offset_head = StateDeltaHead(feature_dim=int(feature_dim),
                                 prop_keys=prop_keys,
                                 n_states=n_states)
    return StateRoutedOffsetSo3krates(
        backbone=backbone,
        offset_head=offset_head,
        prop_keys=prop_keys,
        pretrained_ground_ckpt_dir=pretrained_ground_ckpt_dir,
        ground_checkpoint_step=ground_checkpoint_step,
        ground_checkpoint_fingerprint=ground_checkpoint_fingerprint,
        freeze_pretrained_backbone=freeze_pretrained_backbone,
        bond_backbone_upgrade=bond_backbone_upgrade,
        bond_descriptor_mode=bond_descriptor_mode,
        bond_feature_dim=bond_feature_dim,
        bond_parameter_layout=bond_parameter_layout,
        state_ids=tuple(state_ids),
        required_dataset_keys=tuple(required_dataset_keys))


def init_delta_offset_model(h: Dict,
                            freeze_pretrained_backbone: bool = None) -> StateRoutedOffsetSo3krates:
    if h.get('training_mode') != DELTA_OFFSET_TRAINING_MODE:
        raise ValueError('Checkpoint is not marked with `training_mode: delta_offset`.')
    offset_h = h['delta_offset_model']
    if offset_h.get('training_mode', DELTA_OFFSET_TRAINING_MODE) != DELTA_OFFSET_TRAINING_MODE:
        raise ValueError('Delta-offset model metadata has an incompatible training mode.')

    backbone = init_stack_net({'stack_net': offset_h['backbone']})
    offset_head = StateDeltaHead(**offset_h['offset_head'])
    freeze_backbone = (offset_h.get('freeze_pretrained_backbone', False)
                       if freeze_pretrained_backbone is None else freeze_pretrained_backbone)
    return StateRoutedOffsetSo3krates(
        backbone=backbone,
        offset_head=offset_head,
        prop_keys=offset_h['prop_keys'],
        pretrained_ground_ckpt_dir=offset_h.get('pretrained_ground_ckpt_dir'),
        ground_checkpoint_step=offset_h.get('ground_checkpoint_step'),
        ground_checkpoint_fingerprint=offset_h.get('ground_checkpoint_fingerprint'),
        freeze_pretrained_backbone=freeze_backbone,
        bond_backbone_upgrade=offset_h.get('bond_backbone_upgrade', False),
        bond_descriptor_mode=offset_h.get('bond_descriptor_mode', ABSOLUTE_BOND_DESCRIPTOR),
        bond_feature_dim=offset_h.get('bond_feature_dim', ABSOLUTE_BOND_FEATURE_DIM),
        bond_parameter_layout=offset_h.get('bond_parameter_layout', LEGACY_BOND_LAYOUT),
        state_ids=tuple(offset_h.get('state_ids', (1, 2))),
        offset_sign_convention=offset_h.get('offset_sign_convention', OFFSET_SIGN_CONVENTION),
        required_dataset_keys=tuple(offset_h.get('required_dataset_keys', ())),
        teacher_prediction_mode=offset_h.get('teacher_prediction_mode', 'online_before_optimization'))


def build_delta_offset_targets(active_energy,
                               active_force,
                               pred_energy_0,
                               pred_force_0):
    """Construct active-minus-frozen-teacher targets in raw dataset units."""
    active_energy = jnp.asarray(active_energy)
    active_force = jnp.asarray(active_force)
    pred_energy_0 = jnp.asarray(pred_energy_0)
    pred_force_0 = jnp.asarray(pred_force_0)
    if active_energy.shape != pred_energy_0.shape:
        raise ValueError('Active and teacher energy arrays must have identical shapes.')
    if active_force.shape != pred_force_0.shape:
        raise ValueError('Active and teacher force arrays must have identical shapes.')
    return (active_energy - jax.lax.stop_gradient(pred_energy_0),
            active_force - jax.lax.stop_gradient(pred_force_0))


def reconstruct_delta_offset_predictions(pred_energy_0,
                                         pred_force_0,
                                         offset_energy,
                                         offset_force):
    """Reconstruct active-state predictions from teacher plus learned offset."""
    pred_energy_0 = jnp.asarray(pred_energy_0)
    pred_force_0 = jnp.asarray(pred_force_0)
    offset_energy = jnp.asarray(offset_energy)
    offset_force = jnp.asarray(offset_force)
    if pred_energy_0.shape != offset_energy.shape:
        raise ValueError('Teacher and offset energy arrays must have identical shapes.')
    if pred_force_0.shape != offset_force.shape:
        raise ValueError('Teacher and offset force arrays must have identical shapes.')
    return pred_energy_0 + offset_energy, pred_force_0 + offset_force


def restore_ground_prediction_units(outputs, inputs, scales, prop_keys):
    """Undo ground-target normalization before defining offset targets."""
    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    z_key = prop_keys[pn.atomic_type]
    for quantity in (pn.energy, pn.force):
        if quantity not in scales:
            raise KeyError(f'Ground checkpoint scales are missing `{quantity}`.')

    energy_scale = jnp.asarray(scales[pn.energy]['scale'])
    per_atom_shift = jnp.asarray(scales[pn.energy]['per_atom_shift'])
    z = jnp.asarray(inputs[z_key], dtype=jnp.int32)
    node_mask_key = prop_keys.get(pn.node_mask)
    node_mask = (jnp.asarray(inputs[node_mask_key]).astype(bool)
                 if node_mask_key in inputs else z > 0)
    safe_z = jnp.where(node_mask, z, 0)
    structure_shift = jnp.take(per_atom_shift, safe_z).sum(axis=-1)
    energy = energy_scale * outputs[energy_key] + structure_shift[..., None]
    force = jnp.asarray(scales[pn.force]['scale']) * outputs[force_key]
    return energy, force


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
