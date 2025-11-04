import importlib.resources
from pathlib import Path


def load_data(filename):
    p_filename = Path(filename)
    if p_filename.suffix == '.npz':
        import numpy as np
        stream = importlib.resources.path(__name__, filename)
        return np.load(stream)
    else:
        from ase.io import iread
        f = importlib.resources.path(__name__, filename)
        return iread(f, ':')
