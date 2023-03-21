import utils
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import mplhep as hep
from coffea.util import load
import numpy as np
import pandas as pd
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from torch.utils.data import Dataset
from numba import njit
import vector
import numba as nb
import numpy.ma as ma
vector.register_numba()
vector.register_awkward()

hep.style.use(hep.style.ROOT)


class Dataset_RecoLevel(Dataset):
    def __init__(self, root, object_types=["jets", "lepton_reco", "met", "boost"], transform=None):

        self.fields = {
            "jets": ["pt", "eta", "phi", "btag", "prov"],
            "lepton_reco": ["pt", "eta", "phi", "m"],
            "met": ["pt", "eta", "phi", "m"],
            "boost": ["x", "y", "z", "t"]
        }

        self.root = root
        self.transform = transform
        self.object_types = object_types

        # if an object is missing (example: jets/lepton_reco/met/boost => compute boost)
        for object_type in self.object_types:
            if not os.path.isfile(self.processed_file_names(object_type)):
                print("File missing: compute boost")
                self.boost = self.get_boost()
                break

        for object_type in self.object_types:
            if not os.path.isfile(self.processed_file_names(object_type)):
                print("Create new file for " + object_type)
                self.process(object_type)
            else:
                print(object_type + " file already exists")

        self.mask_jets, self.data_jets = torch.load(
            self.processed_file_names("jets"))
        self.mask_lepton, self.data_lepton = torch.load(
            self.processed_file_names("lepton_reco"))
        self.mask_met, self.data_met = torch.load(
            self.processed_file_names("met"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))

    @property
    def raw_file_names(self):
        return [self.root + '/all_jets_v6.parquet']

    def processed_file_names(self, type):

        return (self.root + '/processed_jets/' + type + '_data.pt')

    def get_boost(self):

        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            jets = df["jets"]
            jets = ak.with_name(jets, name="Momentum4D")

            leptons = df["lepton_reco"]
            leptons = ak.with_name(leptons, name="Momentum4D")

            met = df["met"]
            met = ak.with_name(met, name="Momentum4D")

            boost_jets = utils.get_vector_sum(jets)
            boost = boost_jets + leptons + met

        return boost

    def boost_CM(self, objects_array, boost):
        objects_CM = objects_array.boost_p4(boost.neg3D)

        # Overwriting old pt by calling the function on the boosted object
        objects_CM["pt"] = objects_CM.pt
        objects_CM["eta"] = objects_CM.eta
        objects_CM["phi"] = objects_CM.phi

        return objects_CM

    def Reshape(self, input, value, ax):
        max_no = ak.max(ak.num(input, axis=ax))
        input_padded = ak.pad_none(input, max_no, axis=ax)
        input_filled = ak.fill_none(input_padded, value, axis=ax)

        return input_filled

    # Get mask for object with pt = 0
    def get_mask_pt(self, objects_array):
        return objects_array[:, :, 0] > 1e-5

    def process(self, object_type):

        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            if (object_type == "boost"):
                objects = self.boost

            else:
                objects = ak.with_name(df[object_type], name="Momentum4D")
                objects = self.boost_CM(objects, self.boost)

            if object_type == "jets":
                objects = self.Reshape(objects, utils.struct_jets, 1)

            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "jets":
                d_list = np.transpose(d_list, (0, 2, 1))
                mask = self.get_mask_pt(d_list)

            if (object_type == "lepton_reco" or object_type == "met" or object_type == "boost"):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))

            tensor_data = torch.tensor(d_list, dtype=torch.float)
            tensor_mask = torch.tensor(mask, dtype=torch.float)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))

    def __getitem__(self, index):

        return (self.mask_lepton[index], self.data_lepton[index], self.mask_jets[index],
                self.data_jets[index], self.mask_met[index], self.data_met[index],
                self.mask_boost[index], self.data_boost[index])

    def __len__(self):
        size = len(self.mask_lepton)
        return size
