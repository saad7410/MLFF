# NetCDF dataset builders

Run these commands from the repository root. Pass one or more `.nc` files as
plain positional paths. To add more inputs, add more paths before the options.

The builders infer the internal molecule tag from each filename (for example,
`A03` from `a03-all_grads.nc`). Each filename must contain exactly one
letter-and-number tag, and every input must infer a different tag.

## Bond specs

A bond-spec YAML must be provided with `--bond-specs` when running
`make_so3krateX_dataset`, `make_delta_dataset`, or
`make_delta_offset_dataset`. The repository example is
`examples/example_data/bond_spec_alkenes.yaml`; replace it with your own file
when processing other molecules. Every YAML `mol_id` must match the uppercase
tag inferred from its `.nc` filename.

For delta-offset data, provide S0 and every requested active state for each
molecule. The canonical bond arrays describe the active-state student;
`bond_prob_s0`/`bond_mask_s0` describe the S0 teacher on the same geometry.
`astate` is zero-based, and the trainer creates the offset targets.

## Examples

```bash
python -m examples.preprocessing.make_datasets.from_nc.make_so3krateX_dataset \
  path_to_your_molecule.nc \
  --bond-specs path_to_your_bond_spec \
  --numframes 24 \
  --output output_path
```

```bash
python -m examples.preprocessing.make_datasets.from_nc.make_delta_dataset \
  path_to_your_molecule.nc \
  --bond-specs path_to_your_bond_spec \
  --numframes 24 \
  --output output_path
```

```bash
python -m examples.preprocessing.make_datasets.from_nc.make_delta_offset_dataset \
  path_to_your_molecule.nc \
  --bond-specs path_to_your_bond_spec \
  --numframes 24 \
  --output output_path
```

The examples use `--numframes 24` for quick smoke builds; omit it to process all
available rows. Use `--states ...` for specific electronic states. The
physical-delta builder always uses S0, S1, and S2. Run any command with `--help`
for the remaining options.
