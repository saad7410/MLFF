from typing import Dict, Any
from pathlib import Path
import hashlib
import os
import numpy as np

from orbax.checkpoint import PyTreeCheckpointer, Checkpointer, PyTreeCheckpointHandler
from orbax import checkpoint
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict
import pathlib

__STEP_PREFIX__: str = 'ckpt'


def load_params_from_ckpt_dir(ckpt_dir, step=None):
    try:
        return load_state_from_ckpt_dir(ckpt_dir, step=step)['valid_params']
    except ValueError:
        try:
            loaded_mngr = checkpoint.CheckpointManager(
                pathlib.Path(ckpt_dir).resolve(),
                item_names=('state',),
                item_handlers={'state': checkpoint.StandardCheckpointHandler()},
                options=checkpoint.CheckpointManagerOptions(step_prefix="ckpt"),
            )

            restore_step = loaded_mngr.latest_step() if step is None else int(step)
            mngr_state = loaded_mngr.restore(restore_step)

            state = mngr_state.get('state')

            return state['valid_params']
        except ValueError:
            raise RuntimeError(
                f'Loading model parameters from checkpoint saved at {ckpt_dir} failed. '
                'This error typically occurs if within the ckpt_XXX directory there is another folder. '
                'Consider moving the folder somewhere else.'
            )


def latest_checkpoint_step(ckpt_dir: str) -> int:
    """Return the latest numeric ``ckpt_<step>`` directory."""
    ns = []
    abs_ckpt_dir = Path(ckpt_dir).resolve().absolute()
    for u in os.scandir(abs_ckpt_dir):
        if u.is_dir():
            prefix_n = Path(u).stem.split('_')
            if len(prefix_n) == 2 and prefix_n[0] == __STEP_PREFIX__:
                try:
                    ns.append(int(prefix_n[1]))
                except ValueError:
                    continue
    if not ns:
        raise ValueError(f'No `{__STEP_PREFIX__}_<step>` checkpoint found in {abs_ckpt_dir}.')
    return max(ns)


def load_state_from_ckpt_dir(ckpt_dir: str, step=None):
    # mngr = CheckpointManager(ckpt_dir, __CHECKPOINTERS__, options=CheckpointManagerOptions(step_prefix=__STEP_PREFIX__))
    # return mngr.restore(n)['state']

    abs_ckpt_dir = Path(ckpt_dir).resolve().absolute()
    restore_step = latest_checkpoint_step(abs_ckpt_dir) if step is None else int(step)

    ckptr = Checkpointer(PyTreeCheckpointHandler())
    return ckptr.restore(abs_ckpt_dir / f'{__STEP_PREFIX__}_{restore_step}/state', item=None)


def checkpoint_fingerprint(ckpt_dir: str, step=None, params=None) -> str:
    """Hash the exact teacher step, metadata, scales, and parameter values."""
    abs_ckpt_dir = Path(ckpt_dir).resolve().absolute()
    resolved_step = latest_checkpoint_step(abs_ckpt_dir) if step is None else int(step)
    if params is None:
        params = load_params_from_ckpt_dir(abs_ckpt_dir, step=resolved_step)

    digest = hashlib.sha256()
    digest.update(f'step:{resolved_step}\n'.encode())
    for filename in ('hyperparameters.json', 'scales.json'):
        path = abs_ckpt_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Teacher checkpoint is missing {path}.')
        digest.update(filename.encode())
        digest.update(path.read_bytes())

    flat_params = flatten_dict(unfreeze(freeze(params)))
    for path in sorted(flat_params):
        value = np.asarray(flat_params[path])
        digest.update('/'.join(str(part) for part in path).encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def load_checkpoint_identity(ckpt_dir: str):
    """Load one immutable checkpoint step and return params plus its identity."""
    step = latest_checkpoint_step(ckpt_dir)
    params = load_params_from_ckpt_dir(ckpt_dir, step=step)
    fingerprint = checkpoint_fingerprint(ckpt_dir, step=step, params=params)
    return params, step, fingerprint


def _load_params_from_ckpt_dir(ckpt_dir: str):
    return load_state_from_ckpt_dir(ckpt_dir)['valid_params']
