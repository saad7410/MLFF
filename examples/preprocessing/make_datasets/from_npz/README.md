# NPZ dataset builders

Run these commands from the repository root. Pass one or more `.npz` files as
plain positional paths. To add more inputs, add more paths before the options.

The builders infer the internal molecule tag from each filename (for example,
`A01` from `A01_ethene_dynamic.npz`). Every filename must contain exactly one
letter-and-number tag.

Aggregate archives contain `R`, `z`, `astate`, `energy_all`, and `forces_all`.
Filtered archives contain `R`, `F`, `E`, and `z`, and must include a one-based
`state_N` tag in the filename. CLI and bond-spec states are zero-based. Use
aggregate archives for physical-delta datasets because filtered states are
normally not row-aligned.

NPZ values are preserved as provided; the builders do not convert units.

## Bond specs

A bond-spec YAML must be provided with `--bond-specs` when running
`make_so3krateX_dataset`, `make_delta_dataset`, or
`make_delta_offset_dataset`. Each YAML `mol_id` must match the uppercase tag
inferred from its `.npz` filename.

For delta-offset data, provide S0 and every requested active state for each
molecule. The canonical bond arrays describe the active-state student;
`bond_prob_s0`/`bond_mask_s0` describe the S0 teacher on the same geometry.
`astate` is zero-based, and the trainer creates the offset targets.

## Examples

```bash
python -m examples.preprocessing.make_datasets.from_npz.make_so3krateX_dataset \
  path_to_your_molecule.npz \
  --bond-specs path_to_your_bond_spec \
  --numframes 24 \
  --output output_path
```

```bash
python -m examples.preprocessing.make_datasets.from_npz.make_delta_dataset \
  path_to_your_molecule.npz \
  --bond-specs path_to_your_bond_spec \
  --numframes 24 \
  --output output_path
```

```bash
python -m examples.preprocessing.make_datasets.from_npz.make_delta_offset_dataset \
  path_to_your_molecule.npz \
  --bond-specs path_to_your_bond_spec \
  --numframes 24 \
  --output output_path
```

The examples use `--numframes 24` for quick smoke builds; omit it to process all
available rows. Use `--states ...` for specific electronic states. The
physical-delta builder always uses S0, S1, and S2. Run any command with `--help`
for the remaining options.
