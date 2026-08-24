"""Build a physical-delta dataset from aligned SHNITSEL NPZ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[4]
if __package__ in {None, ""} and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.preprocessing.make_datasets.cli_utils import metadata_overrides
from examples.preprocessing.make_datasets.data_sources import NpzFileSource
from examples.preprocessing.make_datasets.dataset_builders import assemble_delta_dataset


GROUND_STATE = 0
DELTA_STATE_1 = 1
DELTA_STATE_2 = 2


def build_delta_dataset(
    *,
    npz_inputs: Sequence[str | Path],
    output: str | Path,
    numframes: int | None,
    r_cut: float,
    bond_specs_file: str | Path,
    skip_missing_bond_specs: bool,
    check_geometry_alignment: bool,
    geometry_atol: float,
) -> None:
    """Build physical S1-S0 and S2-S0 targets from aligned state archives."""
    source = NpzFileSource.from_paths(npz_inputs)
    assemble_delta_dataset(
        source=source,
        output=output,
        molecules=None,
        numframes=numframes,
        r_cut=r_cut,
        bond_specs_file=bond_specs_file,
        skip_missing_bond_specs=skip_missing_bond_specs,
        check_geometry_alignment=check_geometry_alignment,
        geometry_atol=geometry_atol,
        metadata_overrides=metadata_overrides(
            source,
            source_description="aligned SHNITSEL state NPZ files",
            builder="make_delta_dataset.py",
            selection_logic=(
                "aligned state rows; R/E/F come from S0, Delta_E1/F1 from S1-S0, "
                "and Delta_E2/F2 from S2-S0"
            ),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a delta dataset from aligned NPZ files with Delta_E/F targets "
            "and state-specific bonds."
        )
    )
    parser.add_argument(
        "npz_inputs",
        nargs="+",
        metavar="PATH.npz",
        help="One or more aligned NPZ files; provide each file as a separate path.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--numframes",
        type=int,
        default=None,
        help=(
            "Optional total dataset frame count, divided equally among molecules. "
            "The total is reduced if any molecule cannot provide its share."
        ),
    )
    parser.add_argument(
        "--r-cut",
        type=float,
        default=5.0,
        help="Neighbor cutoff in the same distance unit as the input R arrays.",
    )
    parser.add_argument(
        "--bond-specs",
        required=True,
        metavar="PATH",
        help="Required YAML bond-spec file for every input molecule and state.",
    )
    parser.add_argument("--skip-missing-bond-specs", action="store_true")
    parser.add_argument(
        "--allow-unaligned-delta-rows",
        action="store_true",
        help=(
            "Disable the default geometry-alignment check. Use only if you intentionally "
            "want row-index deltas even when state_1/state_2/state_3 rows are not the "
            "same geometries."
        ),
    )
    parser.add_argument("--geometry-atol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_delta_dataset(
        npz_inputs=args.npz_inputs,
        output=args.output,
        numframes=args.numframes,
        r_cut=args.r_cut,
        bond_specs_file=args.bond_specs,
        skip_missing_bond_specs=args.skip_missing_bond_specs,
        check_geometry_alignment=not args.allow_unaligned_delta_rows,
        geometry_atol=args.geometry_atol,
    )
