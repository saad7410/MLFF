import flax.linen as nn
import jax.numpy as jnp
import json
import os

from pathlib import Path
from typing import (Any, Callable, Dict, Sequence, Tuple)

from mlff.nn.layer import get_layer
from mlff.nn.embed import get_embedding_module
from mlff.nn.observable import get_observable_module
from mlff.io import read_json
from mlff.masking.mask import safe_scale
from mlff.properties import property_names as pn


Array = Any


class StackNet(nn.Module):
    geometry_embeddings: Sequence[Callable]
    feature_embeddings: Sequence[Callable]
    layers: Sequence[Callable]
    observables: Sequence[Callable]
    prop_keys: Dict

    def setup(self):
        if len(self.feature_embeddings) == 0:
            msg = "At least one embedding module in `feature_embeddings` is required."
            raise ValueError(msg)
        if len(self.observables) == 0:
            msg = "At least one observable module in `observables` is required."
            raise ValueError(msg)

    @classmethod
    def create_from_ckpt_dir(cls, ckpt_dir: str):
        h_path = Path(ckpt_dir).absolute().resolve() / 'hyperparameters.json'
        stack_net = init_stack_net(read_json(h_path))
        return stack_net

    @nn.compact
    def __call__(self,
                 inputs,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Energy function of the NN.

        Args:
            inputs (Dict):
            args (Tuple):
            kwargs (Dict):

        Returns: energy, shape: (1)

        """

        # Build the shared representation once before applying each configured observable head.
        quantities = self.forward_features(inputs)

        observables = {}
        for o_fn in self.observables:
            o_dict = o_fn(quantities)
            observables.update(o_dict)

        # Return only model observables while keeping intermediate features private by default.
        return observables

    def forward_features(self, inputs) -> Dict[str, jnp.ndarray]:
        """Return the final atomwise SO3krates quantities without applying observables."""
        quantities = {}
        quantities.update(inputs)

        # Detect bond-aware layers from saved or newly constructed layer metadata.
        bond_aware = any(getattr(layer, 'bond_aware', False) for layer in self.layers)

        # Canonicalize optional custom edge-property keys for the shared layer call convention.
        pair_mask_key = self.prop_keys.get(pn.pair_mask, pn.pair_mask)
        supplied_pair_mask = inputs.get(pair_mask_key)

        # Initialize masks
        quantities.update(init_masks(z=inputs[self.prop_keys['atomic_type']],
                                     idx_i=inputs['idx_i'],
                                     pair_mask=supplied_pair_mask)
                          )

        if bond_aware:
            # Fail immediately when either required bond tensor is absent from a bond-aware model call.
            bond_prob_key = self.prop_keys.get(pn.bond_prob, pn.bond_prob)
            bond_mask_key = self.prop_keys.get(pn.bond_mask, pn.bond_mask)
            if bond_prob_key not in inputs or bond_mask_key not in inputs:
                raise ValueError('Bond-aware SO3krates requires both `bond_prob` and `bond_mask` inputs.')

            # Validate the per-sample edge contract before any layer parameters are evaluated.
            bond_prob = inputs[bond_prob_key]
            bond_mask = inputs[bond_mask_key]
            pair_mask = quantities['pair_mask']
            if bond_prob.ndim != 2 or bond_prob.shape != (pair_mask.shape[0], 4):
                raise ValueError('`bond_prob` must have per-sample shape (P, 4).')
            if bond_mask.ndim != 1 or bond_mask.shape != pair_mask.shape:
                raise ValueError('`bond_mask` must have per-sample shape (P,).')

            # Store canonical names and clear padded descriptors before geometry or attention sees them.
            quantities[pn.bond_prob] = safe_scale(bond_prob, scale=pair_mask[:, None])
            quantities[pn.bond_mask] = safe_scale(bond_mask, scale=pair_mask)

        # Initialize the geometric quantities
        for geom_emb in self.geometry_embeddings:
            geom_quantities = geom_emb(quantities)
            quantities.update(geom_quantities)

        # Initialize the per atom embedding
        embeds = []
        for embed_fn in self.feature_embeddings:
            embeds += [embed_fn(quantities)]  # len: n_embeds, shape: (n,F)
        x = jnp.stack(embeds, axis=-1).sum(axis=-1) / jnp.sqrt(len(embeds))  # shape: (n,F)
        quantities.update({'x': x})

        for (n, layer) in enumerate(self.layers):

            if bond_aware:
                # Reapply padding masks at every layer boundary to keep edge corrections exactly zero.
                quantities[pn.bond_prob] = safe_scale(quantities[pn.bond_prob],
                                                      scale=quantities['pair_mask'][:, None])
                quantities[pn.bond_mask] = safe_scale(quantities[pn.bond_mask],
                                                      scale=quantities['pair_mask'])

            updated_quantities = layer(**quantities)
            quantities.update(updated_quantities)

        # Expose the final invariant and equivariant representation to composite SO3krates models.
        return quantities

    def __dict_repr__(self):
        geometry_embeddings = [x.__dict_repr__() for x in self.geometry_embeddings]
        feature_embeddings = []
        layers = []
        observables = []
        for x in self.feature_embeddings:
            feature_embeddings += [x.__dict_repr__()]
        for (n, x) in enumerate(self.layers):
            layers += [x.__dict_repr__()]
        for x in self.observables:
            observables += [x.__dict_repr__()]

        return {'stack_net': {'geometry_embeddings': geometry_embeddings,
                              'feature_embeddings': feature_embeddings,
                              'layers': layers,
                              'observables': observables,
                              'prop_keys': self.prop_keys,
                              'n_layers': len(layers)}}

    def to_json(self, ckpt_dir, name='hyperparameters.json'):
        j = self.__dict_repr__()
        with open(os.path.join(ckpt_dir, name), 'w', encoding='utf-8') as f:
            json.dump(j, f, ensure_ascii=False, indent=4)

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        self.prop_keys.update(prop_keys)
        if sub_modules:
            all_modules = self.geometry_embeddings + self.feature_embeddings + self.observables
            for m in all_modules:
                m.reset_prop_keys(prop_keys=prop_keys)

    def reset_input_convention(self, input_convention):
        for g in self.geometry_embeddings:
            g.reset_input_convention(input_convention=input_convention)

    def reset_output_convention(self, output_convention):
        for o in self.observables:
            o.reset_output_convention(output_convention=output_convention)


def init_masks(z, idx_i, pair_mask=None):
    point_mask = (z != 0).astype(jnp.float32)  # shape: (n)
    index_pair_mask = (idx_i != -1).astype(jnp.float32)  # shape: (n_pairs)

    if pair_mask is None:
        # Preserve the legacy convention when no explicit precomputed mask is supplied.
        effective_pair_mask = index_pair_mask
    else:
        if pair_mask.ndim != 1 or pair_mask.shape != idx_i.shape:
            raise ValueError('`pair_mask` must have per-sample shape (P,) aligned with `idx_i`.')

        # Never allow a supplied mask to reactivate an index-padded edge.
        effective_pair_mask = pair_mask.astype(jnp.float32) * index_pair_mask

    return {'point_mask': point_mask, 'pair_mask': effective_pair_mask}


def init_stack_net(h) -> StackNet:
    _h = h['stack_net']
    geom_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['geometry_embeddings']]
    feature_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['feature_embeddings']]
    lays = [get_layer(*tuple(x.items())[0]) for x in _h['layers']]
    obs = [get_observable_module(*tuple(x.items())[0]) for x in _h['observables']]
    return StackNet(**{'geometry_embeddings': geom_embs,
                       'feature_embeddings': feature_embs,
                       'layers': lays,
                       'observables': obs,
                       'prop_keys': _h['prop_keys']})
