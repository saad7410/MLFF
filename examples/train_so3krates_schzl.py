import numpy as np
import jax
import jax.numpy as jnp
import os
import wandb
import portpicker
import ase.units as si

from mlff.io.io import create_directory, bundle_dicts, save_dict
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.data import DataTuple, DataSet

from mlff.nn.stacknet import get_obs_and_force_fn, get_observable_fn, get_energy_force_stress_fn
from mlff.nn import So3krates
from mlff.properties import md17_property_keys as prop_keys

import mlff.properties.property_names as pn

# -------------------------------------------------------
# 1. Initialize JAX distributed (single process)
# -------------------------------------------------------
port = portpicker.pick_unused_port()
jax.distributed.initialize(
    f"localhost:{port}",  # address used by JAX
    num_processes=1,      # only one process here
    process_id=0          # this is process 0
)

# -------------------------------------------------------
# 2. Define data path and checkpoint directory
# -------------------------------------------------------
data_path = "example_data/ethanol.npz"  # MD17 ethanol dataset
save_path = "ckpt_dir"

# Folder for this training run: ckpt_dir/module
ckpt_dir = os.path.join(save_path, "module")
ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

# -------------------------------------------------------
# 3. Load MD17 data and convert units
# -------------------------------------------------------
# Get the keys for energy and forces from the MD17 property mapping
E_key = prop_keys["energy"]
F_key = prop_keys["force"]

# Load the .npz file into a dict of numpy arrays
data = dict(np.load(data_path))

# MD17 provides energies and forces in kcal/mol.
# Convert them to eV (and eV/Å) using ASE units.
data[E_key] = data[E_key] * si.kcal / si.mol
data[F_key] = data[F_key] * si.kcal / si.mol

# -------------------------------------------------------
# 4. Wrap data in DataSet, build neighbor lists, and split
# -------------------------------------------------------
r_cut = 5.0  # radial cutoff for neighborhoods in Å

# DataSet holds all arrays and knows how to split and build neighbors
data_set = DataSet(data=data, prop_keys=prop_keys)

# Randomly split into train and validation subsets.
# Here: 200 train samples, 200 validation samples, no explicit test set.
data_set.random_split(
    n_train=200,
    n_valid=200,
    n_test=None,
    mic=False,        # no minimum image convention (no periodic boundary)
    r_cut=r_cut,      # neighbor cutoff
    training=True,    # indicate this is for training
    seed=0            # reproducible split
)

# Center the energy by subtracting its mean (helps training stability).
# The mean and offsets are stored inside the DataSet so they can be undone later.
data_set.shift_x_by_mean_x(x=pn.energy)

# Save the split indices and scaling info for reproducibility
data_set.save_splits_to_file(ckpt_dir, "splits.json")
data_set.save_scales(ckpt_dir, "scales.json")

# Get the actual split data dictionaries:
# d["train"] and d["valid"] are dicts of arrays for each subset.
d = data_set.get_data_split()

# -------------------------------------------------------
# 5. Define the So3krates model
# -------------------------------------------------------
# Small So3krates network:
# - 32 feature channels
# - 2 layers of equivariant message passing
# - degrees [1, 2] for spherical harmonics
# - 2 attention heads
net = So3krates(
    F=32,
    n_layer=2,
    prop_keys=prop_keys,
    geometry_embed_kwargs={
        "degrees": [1, 2],
        "r_cut": r_cut
    },
    so3krates_layer_kwargs={
        "n_heads": 2,
        "degrees": [1, 2]
    }
)

# -------------------------------------------------------
# 6. Create observable function (energy + forces) and optimizer
# -------------------------------------------------------
# Observable function that returns energy and forces for a given batch
obs_fn = get_obs_and_force_fn(net)

# Vectorize obs_fn over batch dimension:
# in_axes=(None, 0) means "params is not batched, inputs are batched on axis 0".
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

# Set up optimizer (e.g., Adam) with a base learning rate of 1e-3
opt = Optimizer()
tx = opt.get(learning_rate=1e-3)

# -------------------------------------------------------
# 7. Configure the Coach: inputs, targets, training loop parameters
# -------------------------------------------------------
coach = Coach(
    # Which properties from the dataset are used as inputs to the network
    inputs=[
        pn.atomic_position,
        pn.atomic_type,
        pn.idx_i,
        pn.idx_j,
        pn.node_mask
    ],
    # Which properties are predicted by the network
    targets=[
        pn.energy,
        pn.force
    ],
    epochs=1000,                 # number of full passes over the training set
    training_batch_size=5,       # mini-batch size for training
    validation_batch_size=5,     # mini-batch size for validation
    # Relative loss weights: emphasize forces (0.99) over energies (0.01)
    loss_weights={
        pn.energy: 0.01,
        pn.force: 0.99
    },
    ckpt_dir=ckpt_dir,           # where to save checkpoints and logs
    data_path=data_path,         # original data path (for logging)
    net_seed=0,                  # random seed for network initialization
    training_seed=0              # random seed for data shuffling, etc.
)

# Combined loss function: wraps obs_fn, applies weights to energy/force errors
loss_fn = get_loss_fn(
    obs_fn=obs_fn,
    weights=coach.loss_weights,
    prop_keys=prop_keys
)

# -------------------------------------------------------
# 8. Build DataTuples for train and validation
# -------------------------------------------------------
# DataTuple knows how to map raw arrays to (inputs, targets) pairs
data_tuple = DataTuple(
    inputs=coach.inputs,
    targets=coach.targets,
    prop_keys=prop_keys
)

# Create indexable datasets for training and validation
train_ds = data_tuple(d["train"])
valid_ds = data_tuple(d["valid"])

# -------------------------------------------------------
# 9. Initialize network parameters and training state
# -------------------------------------------------------
# Take the first training example and convert it into JAX arrays
# This serves as a dummy input for initializing the network parameters
inputs = jax.tree_util.tree_map(
    lambda x: jnp.array(x[0, ...]),
    train_ds[0]
)

# Initialize parameters of the So3krates model
params = net.init(
    jax.random.PRNGKey(coach.net_seed),
    inputs
)

# Create the TrainState, including learning rate schedules
train_state, h_train_state = create_train_state(
    net,
    params,
    tx,
    polyak_step_size=None,  # no Polyak averaging
    plateau_lr_decay={
        "patience": 50,
        "decay_factor": 1.0  # factor 1.0 means "no decay on plateau"
    },
    scheduled_lr_decay={
        # Exponential learning rate decay:
        # every 10,000 steps, LR *= 0.9
        "exponential": {
            "transition_steps": 10_000,
            "decay_factor": 0.9
        }
    }
)

# -------------------------------------------------------
# 10. Collect hyperparameters and dataset info, save to disk
# -------------------------------------------------------
h_net = net.__dict_repr__()       # network config as dict
h_opt = opt.__dict_repr__()       # optimizer config as dict
h_coach = coach.__dict_repr__()   # coach config as dict
h_dataset = data_set.__dict_repr__()

# Merge all configuration dicts into one
h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])

# Save all hyperparameters and settings to hyperparameters.json
save_dict(
    path=ckpt_dir,
    filename="hyperparameters.json",
    data=h,
    exists_ok=True
)

# -------------------------------------------------------
# 11. Initialize Weights & Biases and start training
# -------------------------------------------------------
wandb.init(config=h)

coach.run(
    train_state=train_state,
    train_ds=train_ds,
    valid_ds=valid_ds,
    loss_fn=loss_fn,
    log_every_t=1,      # log every training step
    restart_by_nan=True,  # restart training if NaNs appear
    use_wandb=True      # log metrics to Weights & Biases
)
