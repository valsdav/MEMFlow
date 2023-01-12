import os
import numpy as np
from . import ttH
import numba
import torch

param_card = os.path.join(os.path.dirname(__file__), "ttH", "param_card.dat")

ttH.initialise(param_card)


def smatrix_ttH(pdgs, momenta):
    proc_id = (
        -1
    )  # if you use the syntax "@X" (with X>0), you can set proc_id to that value (this allows to distinguish process with identical initial/final state.)
    nhel = -1  # means sum over all helicity
    scale2 = 0.0  # only used for loop matrix element. should be set to 0 for tree-level
    alphas = 0.13
    # transverse of the momenta for fortran ordering
    return ttH.smatrixhel(pdgs, proc_id, momenta, alphas, scale2, nhel)


def eval_smatrix_batch(pdgs, momenta):
    inputs = np.transpose(momenta.numpy(), (0,2,1))
    output = torch.zeros((momenta.shape[0]))
    for i in range(inputs.shape[0]):
        output[i] = smatrix_ttH(pdgs, inputs[i])
    return output
