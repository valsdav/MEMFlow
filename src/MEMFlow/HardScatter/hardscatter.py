from __future__ import annotations

import numpy as np
import ttH

ttH.initialise("./ttH/param_card.dat")

from numba import jit, vectorize, float64, int32

@vectorize([float64()])
def smatrix_ttH(pdgs, vectors):
    proc_id = (
        -1
    )  # if you use the syntax "@X" (with X>0), you can set proc_id to that value (this allows to distinguish process with identical initial/final state.)
    nhel = -1  # means sum over all helicity
    scale2 = 0.0  # only used for loop matrix element. should be set to 0 for tree-level
    alphas = 0.13
    return ttH.smatrixhel(pdgs, proc_id, vectors, alphas, scale2, nhel)


# smatrix_ttH_many = np.vectorize(smatrix_ttH, otypes=[float])
# out = []
# for i in range(vectors.shape[0]):
#     out.append(smatrix_ttH(pdgs[i], vectors[i]))
# return np.array(out, dtype="float32")
