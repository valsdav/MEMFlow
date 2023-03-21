import hist
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
from coffea.util import load
import numpy as np
import pandas as pd
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from numba import njit
import vector
import numba as nb
import numpy.ma as ma
from torch.utils.data import DataLoader


struct_jets = ak.zip({"pt": np.float32(0),
                      "eta": np.float32(0),
                      "phi": np.float32(0),
                      "btag": np.float32(0),
                      "m": np.float64(0),
                      "matched": bool(0),
                      "prov": 0.},
                     with_name='Momentum4D')

struct_partons = ak.zip({"pt": np.float32(0),
                         "eta": np.float32(0),
                         "phi": np.float32(0),
                         "mass": np.float64(0),
                         "pdgId": bool(0),
                         "prov": 0.},
                        with_name='Momentum4D')


@njit
def sum_vectors_all(arrays):
    size = len(arrays)
    results = np.zeros((size, 4))
    for i, array in enumerate(arrays):
        total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for v in array:
            total = total + v
        results[i, 0] = total.px
        results[i, 1] = total.py
        results[i, 2] = total.pz
        results[i, 3] = total.E
    return results


def get_vector_sum(vectors):
    out = sum_vectors_all(vectors)
    return vector.awk({
        "px": out[:, 0],
        "py": out[:, 1],
        "pz": out[:, 2],
        "E": out[:, 3],
    })


def to_flat_numpy(X, fields, axis=1, allow_missing=False):
    return np.stack([ak.to_numpy(X[f], allow_missing=allow_missing) for f in fields], axis=axis)


def to_flat_tensor(X, fields, axis=1, allow_missing=False):
    return torch.tensor(np.stack([ak.to_numpy(X[f], allow_missing=allow_missing) for f in fields], axis=axis))
