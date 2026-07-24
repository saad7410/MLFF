import jax.numpy as jnp
from jax import lax
from typing import (Callable, Dict, Tuple)

from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from mlff.masking.mask import safe_mask
from mlff.properties import property_names as pn

DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
BatchImportanceWeights = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]


def mse_loss(*, y, y_true): return jnp.mean((y_true - y)**2)


def scaled_safe_masked_mse_loss(y, y_true, scale, msk):
    """

    Args:
        y (): shape: (B,d1, *, dN)
        y_true (): (B,d1, *, dN)
        scale (): (d1, *, dN) or everything broadcast-able to (B, d1, *, dN)
        msk (): shape: (B,*)

    Returns:

    """
    full_mask = ~jnp.isnan(y_true) & msk
    v = safe_mask(full_mask, fn=lambda u: scale * (u - y)**2, operand=y_true)
    den = full_mask.reshape(-1).sum().astype(dtype=v.dtype)
    return safe_mask(den > 0, lambda x: v.reshape(-1).sum() / x, den, 0)


masks = {pn.energy: lambda u: jnp.ones(len(u)).astype(bool)[:, None],
         pn.delta_energy_1: lambda u: jnp.ones(len(u)).astype(bool)[:, None],
         pn.delta_energy_2: lambda u: jnp.ones(len(u)).astype(bool)[:, None],
         pn.offset_energy: lambda u: jnp.ones(len(u)).astype(bool)[:, None],
         pn.atomic_energy: lambda u: u[..., None],
         pn.force: lambda u: u[..., None],
         pn.delta_force_1: lambda u: u[..., None],
         pn.delta_force_2: lambda u: u[..., None],
         pn.offset_force: lambda u: u[..., None],
         pn.stress: lambda u: jnp.ones(len(u)).astype(bool)[:, None, None],
         pn.partial_charge: lambda u: u[..., None],
         pn.hirshfeld_volume: lambda u: u[..., None],
         pn.hirshfeld_volume_ratio: lambda u: u[..., None]
         }


def _stop_gradient_subtrees(params, frozen_param_paths):
    # Flatten once so path prefixes can freeze selected model subtrees without changing optimizer structure.
    flat_params = flatten_dict(unfreeze(freeze(params)))
    for path, value in flat_params.items():
        if any(path[:len(prefix)] == prefix for prefix in frozen_param_paths):
            flat_params[path] = lax.stop_gradient(value)
    return freeze(unflatten_dict(flat_params))


def get_loss_fn(obs_fn: Callable,
                weights: Dict,
                prop_keys: Dict,
                scales: Dict = None,
                frozen_param_paths: Tuple[Tuple[str, ...], ...] = None):
    _weights = {prop_keys[k]: v for (k, v) in weights.items()}
    if scales is None:
        _scales = {prop_keys[k]: jnp.ones(1) for (k, _) in weights.items()}
    else:
        _scales = scales

    _masks = {prop_keys[k]: masks[k] for k in weights.keys()}

    # _with_stress = pn.stress in list(weights.keys())

    def loss_fn(params, batch: DataTupleT):
        inputs, targets = batch

        # Optionally freeze pretrained representation leaves while retaining the existing optimizer/checkpoint tree.
        model_params = (params if frozen_param_paths is None
                        else _stop_gradient_subtrees(params, frozen_param_paths))
        outputs = obs_fn(model_params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():  # name is the value in prop_keys
            _l = scaled_safe_masked_mse_loss(y=outputs[name],
                                             y_true=targets[name],
                                             scale=_scales[name],
                                             msk=_masks[name](inputs[prop_keys[pn.node_mask]])
                                             )

            loss += _weights[name] * _l
            train_metrics.update({name: _l / _scales[name].mean()})

        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics

    return loss_fn


def get_active_learning_loss_fn(obs_fn, weights):
    def loss_fn(params, batch: BatchImportanceWeights):
        inputs, targets, imp_weights = batch  # shape: (B, ...), (B, ...), (B)
        imp_weights_normalized = jnp.sqrt(imp_weights / imp_weights.mean())  # shape: (B)

        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():
            _y = jnp.einsum('b, b... -> b...', imp_weights_normalized, outputs[name])
            _y_true = jnp.einsum('b, b... -> b...', imp_weights_normalized, targets[name])
            _l = mse_loss(y=_y, y_true=_y_true)
            loss += weights[name] * _l
            train_metrics.update({name: _l})
        # for name, w in weights.items():
        #     _l = mse_loss(y_true=outputs[name], y=targets[name])
        #     loss += w * _l
        #     train_metrics.update({name: _l})
        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics
    return loss_fn
