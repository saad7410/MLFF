"""Small shared helpers for the dataset-builder command-line wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOND_SPECS = (
    REPO_ROOT / "examples" / "example_data" / "bond_spec_alkenes.yaml"
)


def metadata_overrides(
    source: Any,
    *,
    source_description: str,
    builder: str,
    selection_logic: str,
    frame_selection_policy: str | None = None,
) -> dict[str, Any]:
    """Combine adapter provenance with the stable, user-facing builder metadata."""
    metadata = dict(source.metadata())
    metadata.update(
        {
            "source": source_description,
            "builder": builder,
            "selection_logic": selection_logic,
        }
    )
    if frame_selection_policy is not None:
        metadata["frame_selection_policy"] = frame_selection_policy
    return metadata
