import logging

import jax.numpy as jnp
import flax.linen as nn

from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Any, Dict

from mlff.nn.observable import StateDeltaHead
from mlff.nn.stacknet.stacknet import StackNet, init_stack_net
from mlff.properties import property_names as pn


class StateSpecificDeltaSo3krates(nn.Module):
    backbone: StackNet
    delta_head: StateDeltaHead
    prop_keys: Dict
    pretrained_ground_ckpt_dir: str = None
    freeze_pretrained_backbone: bool = False

    def setup(self):
        required_outputs = (pn.delta_energy_1, pn.delta_energy_2,
                            pn.delta_force_1, pn.delta_force_2)
        missing_outputs = [name for name in required_outputs if name not in self.prop_keys]
        if missing_outputs:
            raise KeyError(f'Delta SO3krates is missing property mappings for {missing_outputs}.')

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
                                'freeze_pretrained_backbone': self.freeze_pretrained_backbone}}

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        # Keep dataset key overrides synchronized across the wrapper, backbone, and delta head.
        self.prop_keys.update(prop_keys)
        if sub_modules:
            self.backbone.reset_prop_keys(prop_keys=prop_keys)
            self.delta_head.reset_prop_keys(prop_keys=prop_keys)


def init_state_specific_delta_so3krates(backbone: StackNet,
                                        pretrained_ground_ckpt_dir: str = None,
                                        freeze_pretrained_backbone: bool = False,
                                        n_states: int = 3) -> StateSpecificDeltaSo3krates:
    feature_dim = getattr(backbone.feature_embeddings[0], 'features', None)
    if feature_dim is None:
        raise ValueError('The delta head requires a backbone feature embedding with a `features` attribute.')

    # Match the delta readout width to the serialized ground-state representation width.
    prop_keys = dict(backbone.prop_keys)
    delta_head = StateDeltaHead(feature_dim=int(feature_dim),
                                prop_keys=prop_keys,
                                n_states=n_states)
    return StateSpecificDeltaSo3krates(backbone=backbone,
                                       delta_head=delta_head,
                                       prop_keys=prop_keys,
                                       pretrained_ground_ckpt_dir=pretrained_ground_ckpt_dir,
                                       freeze_pretrained_backbone=freeze_pretrained_backbone)


def init_delta_model(h: Dict) -> StateSpecificDeltaSo3krates:
    delta_h = h['delta_model']

    # Reconstruct the shared representation and state head from delta checkpoint metadata.
    backbone = init_stack_net({'stack_net': delta_h['backbone']})
    delta_head = StateDeltaHead(**delta_h['delta_head'])
    return StateSpecificDeltaSo3krates(
        backbone=backbone,
        delta_head=delta_head,
        prop_keys=delta_h['prop_keys'],
        pretrained_ground_ckpt_dir=delta_h.get('pretrained_ground_ckpt_dir'),
        freeze_pretrained_backbone=delta_h.get('freeze_pretrained_backbone', False))


def load_pretrained_backbone(delta_variables,
                             ground_variables,
                             strict: bool = True):
    # Convert both variable trees to mutable dictionaries for a shape-checked representation transfer.
    delta_tree = unfreeze(freeze(delta_variables))
    ground_tree = unfreeze(freeze(ground_variables))
    if 'params' not in delta_tree or 'params' not in ground_tree:
        raise KeyError('Both initialized delta variables and ground checkpoint variables need a `params` collection.')
    if 'backbone' not in delta_tree['params']:
        raise KeyError('Initialized delta variables do not contain the expected `params/backbone` subtree.')

    current_flat = flatten_dict(delta_tree['params']['backbone'])
    ground_flat = flatten_dict(ground_tree['params'])
    matched = {}
    missing = []
    for path, current_value in current_flat.items():
        ground_value = ground_flat.get(path)
        if ground_value is None or jnp.shape(ground_value) != jnp.shape(current_value):
            missing.append(path)
        else:
            matched[path] = ground_value

    if strict and missing:
        preview = ', '.join('/'.join(path) for path in missing[:8])
        raise ValueError(f'Ground checkpoint is incompatible with delta backbone parameters: {preview}.')
    if not matched:
        raise ValueError('Ground checkpoint did not contain any shape-compatible delta backbone parameters.')

    # Replace only compatible backbone leaves and preserve the newly initialized delta-head parameters.
    current_flat.update(matched)
    delta_tree['params']['backbone'] = unflatten_dict(current_flat)
    logging.info('Loaded %d pretrained SO3krates backbone parameter arrays.', len(matched))
    return freeze(delta_tree)
