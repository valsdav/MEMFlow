from memflow.read_data import utils
from memflow.phasespace.phasespace import PhaseSpace
import os
import os.path
import numpy.ma as ma
import numba as nb
from numba import njit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import awkward as ak
import vector
import torch
import numpy as np
torch.set_default_dtype(torch.double)


class Dataset_PartonLevel(Dataset):
    def __init__(self, root, object_types=["partons", "lepton_partons", "boost", "incoming_particles_boost",
                                           "H_t_tbar", "H_t_tbar_cartesian"], transform=None):

        self.fields = {
            "partons": ["pt", "eta", "phi", "mass", "pdgId", "prov"],
            "boost": ["t", "x", "y", "z"],
            "incoming_particles_boost": ["t", "x", "y", "z"],
            "lepton_partons": ["pt", "eta", "phi", "mass", "pdgId"],
            "H_t_tbar": ["pt", "eta", "phi", "mass"],
            "H_t_tbar_cartesian": ["E", "px", "py", "pz"]
        }

        self.root = root
        os.makedirs(self.root + "/processed_partons", exist_ok=True)
        self.transform = transform
        self.object_types = object_types

        (self.partons_boosted, self.leptons_boosted,
         self.higgs_boosted, self.generator,
         self.incoming_particles_boost, self.boost) = self.get_PartonsAndLeptonsAndBoost()

        for object_type in self.object_types:
            if not os.path.isfile(self.processed_file_names(object_type)):
                print("Create new file for " + object_type)
                if object_type == "H_t_tbar":
                    self.process_intermediateParticles()
                elif object_type == "H_t_tbar_cartesian":
                    self.process_intermediateParticles_cartesian()
                else:
                    self.process(object_type)
            else:
                print(object_type + " file already exists")

        self.mask_partons, self.data_partons = torch.load(
            self.processed_file_names("partons"))
        self.mask_lepton_partons, self.data_lepton_partons = torch.load(
            self.processed_file_names("lepton_partons"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))
        self.mask_incoming_particles_boost, self.data_incoming_particles_boost = torch.load(
            self.processed_file_names("incoming_particles_boost"))
        self.data_higgs_t_tbar = torch.load(
            self.processed_file_names("H_t_tbar"))
        self.data_higgs_t_tbar_cartesian = torch.load(
            self.processed_file_names("H_t_tbar_cartesian"))

        self.get_PS_intermediateParticles()
        self.phasespace_intermediateParticles = torch.load(
            self.processed_file_names("phasespace_intermediateParticles"))

    @property
    def raw_file_names(self):
        return [self.root + '/all_jets_v7.parquet']

    def processed_file_names(self, type):
        return (self.root + '/processed_partons/' + type + '_data.pt')

    def get_PartonsAndLeptonsAndBoost(self):
        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            generator = df["generator_info"]
            generator = ak.with_name(generator, name="Momentum4D")

            incoming_particles_boost = self.get_incoming_particles_boost(
                generator)

            partons = df["partons"]
            partons = ak.with_name(partons, name="Momentum4D")

            gluon = partons[partons.prov == 4]
            gluon = self.Reshape(gluon, utils.struct_partons, 1)[:, 0]

            boost = incoming_particles_boost - gluon

            leptons = df["lepton_partons"]
            leptons = ak.with_name(leptons, name="Momentum4D")

            higgs = df["higgs"]
            higgs = ak.with_name(higgs, name="Momentum4D")

            partons_boosted = self.boost_CM(partons, boost)
            leptons_boosted = self.boost_CM(leptons, boost)
            higgs_boosted = self.boost_CM(higgs, boost)

        return partons_boosted, leptons_boosted, higgs_boosted, generator, incoming_particles_boost, boost,

    def get_incoming_particles_boost(self, generator):

        for file in self.raw_file_names:

            x1_numpy = generator.x1.to_numpy()
            x2_numpy = generator.x2.to_numpy()

            pz = (x1_numpy - x2_numpy) * 6500.0
            E = (x1_numpy + x2_numpy) * 6500.0
            zeros = np.zeros(pz.shape)

            boost = ak.Array(
                {
                    "x": zeros,
                    "y": zeros,
                    "z": pz,
                    "t": E,
                }
            )

            boost = ak.with_name(boost, name="Momentum4D")

        return boost

    def boost_CM(self, objects_array, boost):

        objects_CM = objects_array.boost_p4(boost.neg3D)

        # Overwriting old pt by calling the function on the boosted object
        # overwrite objects_array because i dont like "objects_CM.type"
        objects_array["pt"] = objects_CM.pt
        objects_array["eta"] = objects_CM.eta
        objects_array["phi"] = objects_CM.phi

        return objects_array

    def Reshape(self, input, value, ax):
        max_no = ak.max(ak.num(input, axis=ax))
        input_padded = ak.pad_none(input, max_no, axis=ax)
        input_filled = ak.fill_none(input_padded, value, axis=ax)

        return input_filled

    # Get mask for object with pt = 0
    def get_mask_pt(self, objects_array):
        return objects_array[:, :, 0] > 1e-5

    def process(self, object_type):

        # Don't need to boost in CM frame because the partons/leptons are already boosted

        for file in self.raw_file_names:

            if (object_type == "boost"):
                objects = self.boost
            elif (object_type == "incoming_particles_boost"):
                objects = self.incoming_particles_boost
            elif (object_type == "partons"):
                objects = self.partons_boosted
            elif (object_type == "lepton_partons"):
                objects = self.leptons_boosted

            if object_type == "partons":
                objects = self.Reshape(objects, utils.struct_partons, 1)

            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "partons" or object_type == "lepton_partons":
                d_list = np.transpose(d_list, (0, 2, 1))
                if object_type == "partons":
                    mask = self.get_mask_pt(d_list)
                else:
                    mask = np.ones((d_list.shape[0], d_list.shape[1]))

            elif (object_type in ["boost", "incoming_particles_boost"]):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))

            tensor_data = torch.tensor(d_list, dtype=torch.float)
            tensor_mask = torch.tensor(mask, dtype=torch.float)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))

    def process_intermediateParticles(self):
        higgs = self.higgs_boosted
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()

        # Don't need to boost in CM frame because the partons/leptons are already boosted

        intermediate = [higgs, top_hadronic, top_leptonic]

        for i, objects in enumerate(intermediate):
            d_list = utils.to_flat_numpy(
                objects, self.fields["H_t_tbar"], axis=1, allow_missing=False)

            d_list = np.expand_dims(d_list, axis=1)

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        print(intermediate_np.shape)
        tensor_data = torch.tensor(intermediate_np, dtype=torch.float)
        torch.save(tensor_data, self.processed_file_names("H_t_tbar"))

    def process_intermediateParticles_cartesian(self):
        higgs = self.higgs_boosted
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()

        # Don't need to boost in CM frame because the partons/leptons are already boosted
        intermediate = [higgs, top_hadronic, top_leptonic]

        for i, objects in enumerate(intermediate):

            objects_cartesian = self.change_to_cartesianCoordinates(objects)

            d_list = utils.to_flat_numpy(
                objects_cartesian, self.fields["H_t_tbar_cartesian"], axis=1, allow_missing=False)

            d_list = np.expand_dims(d_list, axis=1)

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        print(intermediate_np.shape)
        tensor_data = torch.tensor(intermediate_np, dtype=torch.float)
        torch.save(tensor_data, self.processed_file_names(
            "H_t_tbar_cartesian"))

    def get_PS_intermediateParticles(self):

        E_CM = 13000
        phasespace = PhaseSpace(E_CM, [21, 21], [25, 6, -6])

        incoming_p_boost = self.data_incoming_particles_boost.squeeze()
        x1 = (incoming_p_boost[:, 0] + incoming_p_boost[:, 2]) / 2
        x2 = (incoming_p_boost[:, 0] - incoming_p_boost[:, 2]) / 2

        ps = phasespace.get_ps_from_momenta(
            self.data_higgs_t_tbar_cartesian, x1, x2)

        torch.save(ps, self.processed_file_names(
            "phasespace_intermediateParticles"))

    def change_to_cartesianCoordinates(self, objects):
        px_numpy = objects.px.to_numpy()
        py_numpy = objects.py.to_numpy()
        pz_numpy = objects.pz.to_numpy()
        E_numpy = objects.E.to_numpy()

        objects_cartesian = ak.Array(
            {
                "px": px_numpy,
                "py": py_numpy,
                "pz": pz_numpy,
                "E": E_numpy,
            }
        )

        objects_cartesian = ak.with_name(
            objects_cartesian, name="Momentum4D")

        return objects_cartesian

    def get_Higgs(self):
        partons = self.partons_boosted

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]

        higgs = prov1_partons[:, 0] + prov1_partons[:, 1]

        return higgs

    def get_W_hadronic(self):
        partons = self.partons_boosted

        # find partons with provenance 5 (quarks from W hadronic decay)
        prov5_partons = partons[partons.prov == 5]

        W = prov5_partons[:, 0] + prov5_partons[:, 1]

        return W

    def get_top_hadronic(self):
        partons = self.partons_boosted

        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        W = self.get_W_hadronic()
        top_hadron = W + prov2_partons[:, 0]

        return top_hadron

    def get_W_leptonic(self):
        leptons = self.leptons_boosted

        # sum neutrino and lepton
        W = leptons[:, 0] + leptons[:, 1]

        return W

    def get_top_leptonic(self):
        partons = self.partons_boosted

        W = self.get_W_leptonic()
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]

        top_leptonic = W + prov3_partons[:, 0]

        return top_leptonic

    def get_partons(self):
        return self.partons_boosted

    def __getitem__(self, index):

        return (self.mask_partons[index], self.data_partons[index],
                self.mask_lepton_partons[index], self.data_lepton_partons[index],
                self.mask_boost[index], self.data_boost[index],
                self.mask_incoming_particles_boost[index], self.data_incoming_particles_boost[index],
                self.data_higgs_t_tbar[index], self.data_higgs_t_tbar_cartesian[index],
                self.phasespace_intermediateParticles[index])

    def __len__(self):
        size = len(self.mask_partons)
        return size
