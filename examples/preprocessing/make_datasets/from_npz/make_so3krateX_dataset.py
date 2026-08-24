"""Build a bond-aware So3krateX dataset from SHNITSEL NPZ files."""

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
from examples.preprocessing.make_datasets.dataset_builders import (
    assemble_so3kratex_dataset,
)


def build_so3kratex_dataset(
    *,
    npz_inputs: Sequence[str | Path],
    output: str | Path,
    states: list[int],
    numframes: int | None,
    r_cut: float,
    bond_specs_file: str | Path,
    skip_missing_bond_specs: bool,
) -> None:
    """Build state-expanded, bond-aware rows from one or more NPZ files."""
    source = NpzFileSource.from_paths(npz_inputs)
    assemble_so3kratex_dataset(
        source=source,
        output=output,
        molecules=None,
        states=states,
        numframes=numframes,
        r_cut=r_cut,
        bond_specs_file=bond_specs_file,
        skip_missing_bond_specs=skip_missing_bond_specs,
        metadata_overrides=metadata_overrides(
            source,
            source_description="SHNITSEL NPZ files",
            builder="make_so3krateX_dataset.py",
            selection_logic="state-expanded rows from aggregate or filtered NPZ files",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a bond-aware So3krateX dataset from SHNITSEL NPZ files."
    )
    parser.add_argument(
        "npz_inputs",
        nargs="+",
        metavar="PATH.npz",
        help="One or more NPZ files; provide each file as a separate path.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--numframes",
        type=int,
        default=None,
        help=(
            "Optional total output-row count balanced across inputs and states. "
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_so3kratex_dataset(
        npz_inputs=args.npz_inputs,
        output=args.output,
        states=args.states,
        numframes=args.numframes,
        r_cut=args.r_cut,
        bond_specs_file=args.bond_specs,
        skip_missing_bond_specs=args.skip_missing_bond_specs,
    )
