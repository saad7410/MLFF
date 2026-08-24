# MLFF: SO3krateX and delta-learning extensions

This repository is a research fork of [`thorben-frank/mlff`](https://github.com/thorben-frank/mlff), the original MLFF implementation of the SO3krates transformer developed by J. Thorben Frank and collaborators.

Relative to the `origin/main` baseline, this fork focuses on bond-conditioned SO3krates—called **SO3krateX** here—and two excited-state transfer-learning models: physical delta learning and delta-offset learning. For the original SO3krates quickstart, ASE calculator, molecular dynamics, and general MLFF documentation, see the [upstream README](https://github.com/thorben-frank/mlff#readme).

## What this fork adds

| Mode | Main addition | Training entry point |
| --- | --- | --- |
| SO3krateX | SO3krates conditioned on fixed, state-specific four-channel bond probabilities | `train_so3krates --bond_aware` |
| Physical delta | Shared-backbone models for aligned `S1-S0` and `S2-S0` energy/force corrections | `train_so3krates --delta` |
| Delta-offset | Active-state residuals against a pinned, frozen S0 teacher; the student is always bond-aware | `train_so3krates --delta_offset` |
| Preprocessing | Reusable NetCDF and NPZ builders for ordinary, SO3krateX, physical-delta, and delta-offset datasets | `examples/preprocessing/make_datasets/` |
| Evaluation | Reconstruction-aware evaluators for both delta formulations | `evaluate_delta` and `evaluate_delta_offset` |

SO3krateX is a bond-aware configuration of the existing SO3krates model, not a separate `train_so3krateX` executable or exported model class.

## Model extensions

### SO3krateX

SO3krateX injects an invariant bond descriptor into the radial-spherical filters of both SO3krates attention paths. The fixed descriptor order is:

```text
[single, anti-bonding, double, triple]
```

Bond-aware inputs use a directed, padded graph with `idx_i`, `idx_j`, `pair_mask`, `bond_prob`, and `bond_mask`. The model does not infer bonds at runtime. Dataset loading validates graph alignment, padded `-1` edge indices, masks, finite nonnegative probabilities, normalized annotated rows, channel order, and the graph cutoff when metadata is available.

Train a fresh SO3krateX model with:

```bash
train_so3krates \
  --bond_aware \
  --data_file so3kratex_dataset.npz \
  --n_train 1000 \
  --n_valid 100 \
  --r_cut 5.0 \
  --ckpt_dir so3kratex_module
```

Evaluation reads the bond-aware architecture from `hyperparameters.json` and requires a compatible precomputed graph:

```bash
evaluate \
  --ckpt_dir so3kratex_module \
  --apply_to so3kratex_dataset.npz
```

### Physical delta learning

Physical delta learning requires row-aligned S0, S1, and S2 labels and optimizes:

```text
Delta_E1 = E1 - E0          Delta_F1 = F1 - F0
Delta_E2 = E2 - E0          Delta_F2 = F2 - F0
```

The ground checkpoint supplies the architecture and transferred representation parameters. A learned state embedding and shared delta head distinguish S1 from S2.

```bash
train_so3krates \
  --delta \
  --pretrained_ground_ckpt_dir ground_module \
  --data_file delta_dataset.npz \
  --n_train 1000 \
  --n_valid 100 \
  --ckpt_dir delta_module
```

Reconstruct and evaluate S0, S1, and S2 with:

```bash
evaluate_delta \
  --delta_ckpt_dir delta_module \
  --apply_to delta_dataset.npz \
  --batch_size 1
```

Physical-delta metadata records the ground checkpoint path, not an immutable checkpoint fingerprint. Keep that directory stable: evaluation loads its latest checkpoint step unless `--ground_ckpt_dir` overrides it. The current physical-delta evaluator also skips a final incomplete batch, so choose a batch size that divides the selected row count or use `--batch_size 1` when every row must be evaluated.

### Delta-offset learning

Delta-offset learning is intended for active-state S1/S2 rows that do not have aligned S0/S1/S2 labels. A pinned ground checkpoint predicts the S0 baseline on each active geometry, and the trainer forms:

```text
Offset_E = E_active - E_teacher,S0
Offset_F = F_active - F_teacher,S0
```

The frozen teacher identity, including its checkpoint step and fingerprint, is stored with the offset checkpoint. Reconstruction adds the learned offset back to that teacher prediction.

Every delta-offset student follows the SO3krateX descriptor contract:

- every backbone layer is bond-aware;
- canonical `bond_prob`/`bond_mask` contain the active-state four-channel descriptor;
- `bond_prob_s0`/`bond_mask_s0` are routed to the frozen teacher when that teacher is bond-aware;
- geometry-only, relative-descriptor, and routed legacy offset checkpoints are rejected.

```bash
train_so3krates \
  --delta_offset \
  --pretrained_ground_ckpt_dir ground_module \
  --data_file delta_offset_dataset.npz \
  --n_train 1000 \
  --n_valid 100 \
  --ckpt_dir delta_offset_module
```

`--bond_aware` is not needed in this command: delta-offset students enable the four-channel bond branches automatically.

Evaluate active-state reconstruction with:

```bash
evaluate_delta_offset \
  --delta_offset_ckpt_dir delta_offset_module \
  --apply_to delta_offset_dataset.npz
```

## Dataset preparation

The canonical builders live under:

- [NetCDF builders](examples/preprocessing/make_datasets/from_nc/README.md)
- [NPZ builders](examples/preprocessing/make_datasets/from_npz/README.md)

Both interfaces accept one or more plain positional paths, so inputs look like `A03.nc I01.nc ...` or `A03_data.npz I01_data.npz ...`. Users do not assign paths with `A03=...` or `I01=...`. A unique uppercase molecule tag is inferred from each filename only to match dataset provenance and the bond-spec YAML.

The current builder names contain no `_v2` suffix:

- `make_so3krates_dataset.py` builds descriptor-free SO3krates data;
- `make_so3krateX_dataset.py` builds state-expanded SO3krateX data;
- `make_delta_dataset.py` builds aligned physical-delta data;
- `make_delta_offset_dataset.py` builds active-state delta-offset data.

The three bond-aware builders require `--bond-specs`. Supply a YAML entry for every required molecule/state pair; `mol_id` must match the uppercase tag inferred from the source filename. See the [example bond specification](examples/example_data/bond_spec_alkenes.yaml).


Replace `from_nc` with `from_npz` and pass appropriately named NPZ archives for the NPZ workflow. Aggregate NPZ archives are preferred for physical delta because separately filtered state files are normally not row-aligned. NPZ builders preserve values as provided and `--r-cut` uses the same distance unit as `R`; the trainer and evaluators convert values only when `--units` is explicitly supplied. Keep teacher, active, and delta data physically consistent and repeat the same explicit unit mapping during evaluation. The NetCDF reader converts declared source units to Å, eV, and eV/Å and records the conversion metadata.

Each builder writes the training NPZ and adjacent JSON provenance, including source files, graph cutoff, channel order, state selection, and descriptor semantics.

## Installation

The current `pyproject.toml` requires Python 3.12 or newer. Clone this fork and select the editable install matching your hardware:

```bash
git clone --branch delta-learning https://github.com/saad7410/MLFF.git
cd MLFF
python -m pip install --upgrade pip
```

For CPU:

```bash
python -m pip install -e ".[cpu]"
```

For CUDA 12:

```bash
python -m pip install -e ".[cu12]"
```

Check the current [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for driver and accelerator compatibility.

## Current constraints

- Bond-aware models consume fixed, precomputed NPZ graphs and currently support only nonperiodic data.
- The model cutoff must match the cutoff used by the dataset builder.
- Bond annotations come from the supplied YAML; there is no runtime bond discovery.

## Tests

Run the focused delta-offset descriptor contract tests with:

```bash
pytest -q tests/test_delta_offset_descriptor_contract.py
```

The upstream test suite remains under `tests/`.

## Citation

The SO3krates implementation in this fork is built on the original MLFF work. If you use it, cite the corresponding upstream papers:

```bibtex
@article{frank2022so3krates,
  title={So3krates: Equivariant attention for interactions on arbitrary length-scales in molecular systems},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={29400--29413},
  year={2022}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024}
}
```

## License

This fork retains the upstream [MIT License](LICENSE.md), copyright © 2023 J. Thorben Frank.
