from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import os
from pprint import pprint
import torch
import hist
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
vector.register_numba()
vector.register_awkward()


hep.style.use(hep.style.ROOT)


class ConditioningTransformerLayer():
    def __init__(self, jets_features, lepton_features, out_features, nhead, no_layers):
        super().__init__()

        self.lin_jet = nn.Linear(in_features=jets_features,
                                 out_features=out_features - 1, dtype=torch.float32)
        self.lin_lept = nn.Linear(in_features=lepton_features,
                                  out_features=out_features - 1, dtype=torch.float32)
        self.lin_met = nn.Linear(in_features=lepton_features,
                                 out_features=out_features - 1, dtype=torch.float32)
        self.lin_boost = nn.Linear(in_features=lepton_features,
                                   out_features=out_features - 1, dtype=torch.float32)

        self.gelu = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_features,
                                                   nhead=nhead,
                                                   batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=no_layers)

    def reset_parameters(self):
        self.lin_jet.reset_parameters()
        self.lin_lept.reset_parameters()
        self.lin_met.reset_parameters()
        self.lin_boost.reset_parameters()

    def forward(self, batch_jet, batch_lepton, batch_met, batch_boost, mask_jets, mask_lepton, mask_met, mask_boost):

        jets_afterLin = self.gelu(self.lin_jet(
            batch_jet) * mask_jets[:, :, None])
        lept_afterLin = self.gelu(self.lin_lept(batch_lepton))
        met_afterLin = self.gelu(self.lin_met(batch_met))
        boost_afterLin = self.gelu(self.lin_boost(batch_boost))

        batch_size = batch_jet.size(0)
        no_jets = batch_jet.size(1)
        no_lept = batch_lepton.size(1)

        ones = torch.ones(batch_size, no_jets, 1)  # type jet = 1
        two = 2 * torch.ones(batch_size, no_lept, 1)  # type lepton = 2
        three = 3 * torch.ones(batch_size, 1, 1)  # type met = 3
        four = 4 * torch.ones(batch_size, 1, 1)  # type boost = 4

        jet_afterLin_andLabel = torch.cat((jets_afterLin, ones), dim=-1)
        lept_afterLin_andLabel = torch.cat((lept_afterLin, two), dim=-1)
        met_afterLin_andLabel = torch.cat((met_afterLin, three), dim=-1)
        boost_afterLin_andLabel = torch.cat((boost_afterLin, four), dim=-1)

        transformer_input = torch.concat(
            (boost_afterLin_andLabel, lept_afterLin_andLabel, met_afterLin_andLabel, jet_afterLin_andLabel), dim=1)
        transformer_mask = torch.concat(
            (mask_boost, mask_lepton, mask_met, mask_jets), dim=1)

        transformer_output = self.transformer_encoder(
            transformer_input, src_key_padding_mask=transformer_mask == 0)

        # `computing ther average of not masked objects`
        transformer_output_sum = torch.sum(
            transformer_output * torch.unsqueeze(transformer_mask, -1), dim=1)
        N_valid_objects = torch.sum(transformer_mask, dim=1)

        conditional_input = transformer_output_sum / N_valid_objects

        return torch.cat((batch_boost[:, :, 3:4].squeeze(1), conditional_input), dim=1)
