import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.preprocessing.make_datasets.dataset_builders import (
    assemble_delta_offset_dataset,
)
from examples.preprocessing.make_datasets.preprocessing_helpers import StateData
from mlff.data import DataSet
from mlff.nn.representation.delta import ground_teacher_inputs, init_delta_offset_model
from mlff.properties import delta_offset_property_keys, md17_property_keys
from mlff.properties import property_names as pn


class _TwoStateSource:
    def __init__(self):
        self.states = {state: self._state_data(state) for state in (1, 2)}

    @staticmethod
    def _state_data(state):
        return StateData(
            mol_id="A01",
            state_label=state,
            path=Path(f"A01_state_{state + 1}.npz"),
            R=np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
            F=np.full((1, 2, 3), state, dtype=np.float32),
            E=np.asarray([10.0 + state], dtype=np.float32),
            z_per_frame=np.asarray([[6, 6]], dtype=np.int64),
            node_mask=np.ones((1, 2), dtype=bool),
            frame_idx=np.asarray([100 + state], dtype=np.int64),
        )

    def discover_molecule_ids(self, requested=None):
        if requested is None:
            return ["A01"]
        if requested != ["A01"]:
            raise FileNotFoundError(requested)
        return list(requested)

    def load_state(self, mol_id, state_label, *, active_only=False):
        assert mol_id == "A01"
        assert active_only
        return self.states[state_label]

    def metadata(self):
        return {"source_type": "test"}


def _write_bond_specs(path, states=(0, 1, 2)):
    bond_orders = {0: 1.0, 1: 2.0, 2: 3.0}
    payload = {
        "format": "delta4mlff.testing.bond_specs.v1",
        "bond_prob_channels": ["single", "aromatic", "double", "triple"],
        "specs": [
            {
                "mol_id": "A01",
                "state": state,
                "z": [6, 6],
                "bond_idx_i": [0, 1],
                "bond_idx_j": [1, 0],
                "bond_prob": [bond_orders[state], bond_orders[state]],
            }
            for state in states
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_delta_offset_builder_emits_active_and_s0_descriptors(tmp_path):
    bond_specs = tmp_path / "bond_specs.yaml"
    output = tmp_path / "delta_offset.npz"
    _write_bond_specs(bond_specs)

    assemble_delta_offset_dataset(
        source=_TwoStateSource(),
        output=output,
        molecules=None,
        states=[1, 2],
        numframes=None,
        r_cut=2.0,
        bond_specs_file=bond_specs,
        skip_missing_bond_specs=False,
    )

    with np.load(output) as archive:
        descriptor_keys = {
            key for key in archive.files if key.startswith("bond_prob") or key.startswith("bond_mask")
        }
        assert descriptor_keys == {
            "bond_prob",
            "bond_mask",
            "bond_prob_s0",
            "bond_mask_s0",
        }
        np.testing.assert_array_equal(archive["astate"], [1, 2])
        np.testing.assert_array_equal(archive["target_state"], [1, 2])
        np.testing.assert_allclose(archive["bond_prob"][0, :, 2], 1.0)
        np.testing.assert_allclose(archive["bond_prob"][1, :, 3], 1.0)
        np.testing.assert_allclose(archive["bond_prob_s0"][:, :, 0], 1.0)
        assert not {"Offset_E", "Offset_F", "Delta_E1", "Delta_F1"}.intersection(
            archive.files
        )

        prop_keys = {**md17_property_keys, **delta_offset_property_keys}
        graph_data = {
            key: archive[key]
            for key in (
                "z",
                "idx_i",
                "idx_j",
                "pair_mask",
                "bond_prob",
                "bond_mask",
                "bond_prob_s0",
                "bond_mask_s0",
            )
        }

    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["training_mode"] == "delta_offset"
    assert metadata["canonical_bond_state"] == "active_state"
    assert metadata["states"] == [1, 2]
    assert "active state" in metadata["canonical_bond_keys"]
    assert "S0 bonds" in metadata["ground_teacher_bond_keys"]

    dataset = DataSet(prop_keys=prop_keys, data=graph_data, graph_metadata=metadata)
    dataset._validate_precomputed_graph(r_cut=2.0)


def test_delta_offset_builder_requires_s0_bond_spec(tmp_path):
    bond_specs = tmp_path / "active_only.yaml"
    _write_bond_specs(bond_specs, states=(1,))

    with pytest.raises(KeyError, match="missing bond specs for zero-based states \\[0\\]"):
        assemble_delta_offset_dataset(
            source=_TwoStateSource(),
            output=tmp_path / "unused.npz",
            molecules=None,
            states=[1],
            numframes=None,
            r_cut=2.0,
            bond_specs_file=bond_specs,
            skip_missing_bond_specs=False,
        )


def test_delta_offset_builder_requires_a_bond_spec_file(tmp_path):
    with pytest.raises(ValueError, match="bond_specs_file is required"):
        assemble_delta_offset_dataset(
            source=_TwoStateSource(),
            output=tmp_path / "unused.npz",
            molecules=None,
            states=[1],
            numframes=None,
            r_cut=2.0,
            bond_specs_file=None,
            skip_missing_bond_specs=False,
        )


def test_ground_teacher_receives_s0_descriptor_without_mutating_student_inputs():
    prop_keys = {**md17_property_keys, **delta_offset_property_keys}
    active_prob = np.asarray([[[0.0, 0.0, 1.0, 0.0]]], dtype=np.float32)
    active_mask = np.asarray([[True]])
    ground_prob = np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    ground_mask = np.asarray([[True]])
    inputs = {
        prop_keys[pn.bond_prob]: active_prob,
        prop_keys[pn.bond_mask]: active_mask,
        prop_keys[pn.bond_prob_s0]: ground_prob,
        prop_keys[pn.bond_mask_s0]: ground_mask,
        prop_keys[pn.atomic_position]: np.zeros((1, 2, 3), dtype=np.float32),
    }

    teacher_inputs = ground_teacher_inputs(inputs, prop_keys, teacher_bond_aware=True)

    assert teacher_inputs is not inputs
    assert teacher_inputs[prop_keys[pn.bond_prob]] is ground_prob
    assert teacher_inputs[prop_keys[pn.bond_mask]] is ground_mask
    assert inputs[prop_keys[pn.bond_prob]] is active_prob
    assert inputs[prop_keys[pn.bond_mask]] is active_mask
    assert ground_teacher_inputs(inputs, prop_keys, teacher_bond_aware=False) is inputs


@pytest.mark.parametrize("descriptor_mode", [None, "relative_to_s0"])
def test_offset_checkpoint_requires_explicit_absolute_descriptor_mode(descriptor_mode):
    offset_model = {
        "bond_feature_dim": 4,
        "backbone": {
            "layers": [{"so3krates_layer": {"bond_aware": True}}],
        },
    }
    if descriptor_mode is not None:
        offset_model["bond_descriptor_mode"] = descriptor_mode

    with pytest.raises(ValueError, match="bond_descriptor_mode=absolute_state"):
        init_delta_offset_model(
            {
                "training_mode": "delta_offset",
                "delta_offset_model": offset_model,
            }
        )


def test_offset_checkpoint_requires_explicit_four_channel_descriptor_width():
    with pytest.raises(ValueError, match="bond_feature_dim: 4"):
        init_delta_offset_model(
            {
                "training_mode": "delta_offset",
                "delta_offset_model": {
                    "bond_descriptor_mode": "absolute_state",
                    "backbone": {
                        "layers": [{"so3krates_layer": {"bond_aware": True}}],
                    },
                },
            }
        )


def test_offset_checkpoint_rejects_geometry_only_backbone_before_reconstruction():
    with pytest.raises(ValueError, match="fully bond-aware SO3krates backbone"):
        init_delta_offset_model(
            {
                "training_mode": "delta_offset",
                "delta_offset_model": {
                    "bond_descriptor_mode": "absolute_state",
                    "bond_feature_dim": 4,
                    "backbone": {
                        "layers": [{"so3krates_layer": {"bond_aware": False}}],
                    },
                },
            }
        )
