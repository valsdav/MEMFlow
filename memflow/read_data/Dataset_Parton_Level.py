import utils
import os
import os.path
import mplhep as hep
import numpy.ma as ma
import numba as nb
from numba import njit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import awkward as ak
import vector
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
torch.set_default_dtype(torch.double)

# i want to get the two tops - one from semileptonic decay
#  add partons and lepton partons
# prov = 2
#


class Dataset_PartonLevel(Dataset):
    def __init__(self, root, object_types=["partons", "boost"], transform=None):

        self.fields = {
            "partons": ["pt", "eta", "phi", "mass", "pdgId", "prov"],
            "boost": ["x", "y", "z", "t"]
        }

        self.root = root
        self.transform = transform
        self.object_types = object_types

        # if an object is missing (example: partons/boost => compute boost)
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

        self.mask_partons, self.data_partons = torch.load(
            self.processed_file_names("partons"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))

    @property
    def raw_file_names(self):
        return [self.root + '/all_jets_v6.parquet']

    def processed_file_names(self, type):

        return (self.root + '/processed_partons/' + type + '_data.pt')

    def get_boost(self):

        for file in self.raw_file_names:
            df = ak.from_parquet(file)
            generator = df["generator_info"]

            x1_numpy = generator.x1.to_numpy()
            x2_numpy = generator.x2.to_numpy()

            pz = (x1_numpy - x2_numpy) * 6500.0
            E = (x1_numpy + x2_numpy) * 6500.0
            zeros = np.zeros(pz.shape)

            boost = vector.array(
                {
                    "x": zeros,
                    "y": zeros,
                    "z": pz,
                    "t": E,
                }
            )

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

            if object_type == "partons":
                objects = self.Reshape(objects, utils.struct_partons, 1)

            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "partons":
                d_list = np.transpose(d_list, (0, 2, 1))
                mask = self.get_mask_pt(d_list)

            elif (object_type == "boost"):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))

            tensor_data = torch.tensor(d_list, dtype=torch.float)
            tensor_mask = torch.tensor(mask, dtype=torch.float)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))

    def __getitem__(self, index):

        return (self.mask_partons[index], self.data_partons[index],
                self.mask_boost[index], self.data_boost[index])

    def __len__(self):
        size = len(self.mask_partons)
        return size
