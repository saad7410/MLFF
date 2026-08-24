from typing import (Any, Dict, Sequence, Tuple)
from dataclasses import dataclass
Array = Any
DataTupleT = Tuple[Dict[str, Array], Dict[str, Any]]


def select_data_for_model(data: Dict[str, Array],
                          inputs: Sequence[str],
                          targets: Sequence[str],
                          prop_keys: Dict[str, str]) -> Dict[str, Array]:
    # Deduplicate semantic properties while retaining their declared input/target order.
    properties = tuple(dict.fromkeys((*inputs, *targets)))

    # Resolve semantic properties before touching arrays so custom runtime keys remain supported.
    missing_mappings = [name for name in properties if name not in prop_keys]
    if missing_mappings:
        raise KeyError(f'Missing property-key mappings for model quantities {missing_mappings}.')
    selected_keys = tuple(prop_keys[name] for name in properties)

    # Fail at the NPZ boundary instead of silently constructing an incomplete training dataset.
    missing_data = [key for key in selected_keys if key not in data]
    if missing_data:
        raise KeyError(f'Missing model input/target arrays {missing_data}.')

    # Exclude provenance arrays so generic dataset shape correction cannot expand per-frame metadata.
    return {key: data[key] for key in selected_keys}


@dataclass
class DataTuple:
    inputs: Sequence[str]
    targets: Sequence[str]
    prop_keys: Dict[str, str]

    def __post_init__(self):
        self.input_keys = [self.prop_keys[i] for i in self.inputs]
        self.target_keys = [self.prop_keys[t] for t in self.targets]
        self.get_args = lambda data, args: {k: v for (k, v) in data.items() if k in args}

    def __call__(self, ds: Dict[str, Array]) -> DataTupleT:
        inputs = self.get_args(ds, self.input_keys)
        targets = self.get_args(ds, self.target_keys)
        return inputs, targets
