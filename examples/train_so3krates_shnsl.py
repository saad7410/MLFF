from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import wandb
import portpicker

from mlff.io.io import create_directory, bundle_dicts, save_dict
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.data import DataTuple, DataSet

from mlff.nn.stacknet import get_obs_and_force_fn
from mlff.nn import So3krates

from mlff.properties import md17_property_keys as prop_keys
import mlff.properties.property_names as pn

from examples.preprocessing.schnitsel_preprocessor import ShnitselPreprocessor


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = json.load(f)
    return cfg


def filter_by_state(data: dict, state_index: int | None) -> dict:
    """
    Filter frames in the NPZ data by a chosen active state.

    Assumes:
        data["astate"] has shape (n_frames,)
        data["R"], data["F"], data["E"] have shape (n_frames, ...)
        data["z"] or data["Z"] has shape (n_atoms,)  (no filtering needed)
    """
    if state_index is None:
        return data

    astate = data.get("astate", None)
    if astate is None:
        raise KeyError("NPZ data does not contain 'astate' but state_index was specified.")

    astate = np.asarray(astate)
    mask = (astate == state_index)
    n_sel = int(mask.sum())
    print(f"[filter_by_state] state_index={state_index}, selected {n_sel} frames out of {astate.shape[0]}.")

    if n_sel == 0:
        raise RuntimeError(
            f"No frames with astate == {state_index} were found. "
            "Check your config or the NC file."
        )

    filtered = {}
    for k, v in data.items():
        arr = np.asarray(v)

        # Filter only arrays where first dimension is n_frames
        if arr.ndim > 0 and arr.shape[0] == astate.shape[0]:
            filtered[k] = arr[mask, ...]
        else:
            filtered[k] = arr

    return filtered


def build_ckpt_dir(cfg: dict) -> str:
    ckpt_root = Path(cfg["logging"]["ckpt_root"])
    exp_name = cfg["logging"]["experiment_name"]
    ckpt_dir = ckpt_root.joinpath(exp_name).absolute().resolve()
    ckpt_dir = create_directory(ckpt_dir.as_posix(), exists_ok=False)
    return ckpt_dir


def main() -> None:
    # ----------------------------------------------------------------------
    # 1. Parse arguments and load config
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train So3krates on Shnitsel .nc trajectories using a JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    # ----------------------------------------------------------------------
    # 2. Initialize JAX distributed (single process)
    # ----------------------------------------------------------------------
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(
        f"localhost:{port}",
        num_processes=1,
        process_id=0
    )

    # ----------------------------------------------------------------------
    # 3. Preprocess NC → NPZ using ShnitselPreprocessor
    # ----------------------------------------------------------------------
    nc_path = cfg["data"]["nc_path"]
    state_index = cfg["data"].get("state_index")
    r_cut = float(cfg["data"].get("r_cut", 5.0))

    print(f"[main] Using NC file: {nc_path}")
    prep = ShnitselPreprocessor(nc_path)


    # Export a single NPZ 
    npz_path = prep.export_npz(z_key="z", include_all_states=True)
    print(f"[main] Preprocessed NPZ written to: {npz_path}")

    # ----------------------------------------------------------------------
    # 4. Load NPZ and filter by state_index
    # ----------------------------------------------------------------------
    raw_data = dict(np.load(npz_path.as_posix(), allow_pickle=False))
    data = filter_by_state(raw_data, state_index=state_index)

    core_keys = {
        prop_keys[pn.atomic_position],  
        prop_keys[pn.force],            
        prop_keys[pn.energy],           
        prop_keys[pn.atomic_type],    
    }   

    print(f"[main] Core keys expected from NPZ: {core_keys}")
    print(f"[main] Available keys in NPZ: {set(data.keys())}")  

    clean_data = {k: data[k] for k in core_keys if k in data}
    missing = core_keys - set(clean_data.keys())
    if missing:
        raise KeyError(f"Missing required keys in NPZ: {missing}")

    data = clean_data

    # ----------------------------------------------------------------------
    # 5. Build DataSet and splits
    # ----------------------------------------------------------------------
    n_train = cfg["data"].get("n_train", None)
    n_valid = cfg["data"].get("n_valid", None)
    n_test = cfg["data"].get("n_test", None)

    ckpt_dir = build_ckpt_dir(cfg)
    print(f"[main] Checkpoints and logs will be stored in: {ckpt_dir}")

    data_set = DataSet(data=data, prop_keys=prop_keys)

    data_set.random_split(
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        mic=False,
        r_cut=r_cut,
        training=True,
        seed=cfg["training"].get("random_seed", 0)
    )

    data_set.shift_x_by_mean_x(x=pn.energy)

    data_set.save_splits_to_file(ckpt_dir, "splits.json")
    data_set.save_scales(ckpt_dir, "scales.json")

    d = data_set.get_data_split()

    # ----------------------------------------------------------------------
    # 6. Build So3krates model from config
    # ----------------------------------------------------------------------
    features = int(cfg["model"]["features"])
    n_layers = int(cfg["model"]["n_layers"])
    degree = int(cfg["model"]["degree"])
    n_heads = int(cfg["model"]["n_heads"])

    # Ensure n_heads divides features 
    n_heads = max(1, min(features, n_heads))
    for candidate in range(n_heads, 0, -1):
        if features % candidate == 0:
            n_heads = candidate
            break

    degrees = list(range(1, degree + 1))

    sphc_norm = float(cfg["model"].get("sphc_normalization", 1.0))


    net = So3krates(
        F=features,
        n_layer=n_layers,
        prop_keys=prop_keys,
        geometry_embed_kwargs={
            "degrees": degrees,
            "r_cut": r_cut,
            "sphc_normalization": sphc_norm,
        },
        so3krates_layer_kwargs={
            "n_heads": n_heads,
            "degrees": degrees
        }
    )

    # ----------------------------------------------------------------------
    # 7. Observables, optimizer, coach, loss function
    # ----------------------------------------------------------------------
    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

    opt = Optimizer()
    learning_rate = float(cfg["training"]["learning_rate"])
    tx = opt.get(learning_rate=learning_rate)

    epochs = int(cfg["training"]["n_epochs"])
    batch_size = int(cfg["training"]["batch_size"])

    coach = Coach(
        inputs=[
            pn.atomic_position,
            pn.atomic_type,
            pn.idx_i,
            pn.idx_j,
            pn.node_mask
        ],
        targets=[pn.energy, pn.force],
        epochs=epochs,
        training_batch_size=batch_size,
        validation_batch_size=batch_size,
        # make these configurable 
        loss_weights={pn.energy: 0.01, pn.force: 0.99},
        ckpt_dir=ckpt_dir,
        data_path=str(nc_path),
        net_seed=cfg["training"].get("random_seed", 0),
        training_seed=cfg["training"].get("random_seed", 0)
    )

    loss_fn = get_loss_fn(
        obs_fn=obs_fn,
        weights=coach.loss_weights,
        prop_keys=prop_keys
    )

    data_tuple = DataTuple(
        inputs=coach.inputs,
        targets=coach.targets,
        prop_keys=prop_keys
    )

    train_ds = data_tuple(d["train"])
    valid_ds = data_tuple(d["valid"])

    # ----------------------------------------------------------------------
    # 8. Initialize network parameters and training state (with LR schedule)
    # ----------------------------------------------------------------------
    # Take a single example batch for shape inference
    inputs_example = jax.tree_util.tree_map(
        lambda x: jnp.array(x[0, ...]),
        train_ds[0]
    )

    params = net.init(
        jax.random.PRNGKey(coach.net_seed),
        inputs_example
    )

    transition_steps = int(cfg["training"].get("transition_steps", 10_000))
    decay_factor = float(cfg["training"].get("decay_factor", 0.9))
    warmup_steps = cfg["training"].get("warmup_steps", None)
    warmup_init_value = float(cfg["training"].get("warmup_init_value", 0.01))

    lr_warmup = None
    if warmup_steps is not None:
        lr_warmup = {
            "init_value": warmup_init_value,
            "peak_value": 1.0,
            "warmup_steps": int(warmup_steps),
        }

    train_state, h_train_state = create_train_state(
        net,
        params,
        tx,
        polyak_step_size=None,
        plateau_lr_decay={"patience": 50, "decay_factor": 1.0},
        scheduled_lr_decay={
            "exponential": {
                "transition_steps": transition_steps,
                "decay_factor": decay_factor
            }
        },
        lr_warmup=lr_warmup,
    )

    # ----------------------------------------------------------------------
    # 9. Save hyperparameters and initialize WandB
    # ----------------------------------------------------------------------
    h_net = net.__dict_repr__()
    h_opt = opt.__dict_repr__()
    h_coach = coach.__dict_repr__()
    h_dataset = data_set.__dict_repr__()

    h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])

    save_dict(
        path=ckpt_dir,
        filename="hyperparameters.json",
        data=h,
        exists_ok=True
    )

    wandb.init(config=h, name=cfg["logging"]["experiment_name"])

    # ----------------------------------------------------------------------
    # 10. Run training
    # ----------------------------------------------------------------------
    coach.run(
        train_state=train_state,
        train_ds=train_ds,
        valid_ds=valid_ds,
        loss_fn=loss_fn,
        log_every_t=1,
        restart_by_nan=True,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
