from memflow.read_data import utils
from memflow.phasespace.phasespace import PhaseSpace
import os
import numpy.ma as ma
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import awkward as ak

import torch
import numpy as np
import hist
from numba import njit
#torch.set_default_dtype(torch.double)

from .utils import get_weight

M_HIGGS = 125.25
M_TOP = 172.5
M_GLUON = 1e-3

class Dataset_PartonLevel(Dataset):
    def __init__(self, root, object_types=["partons", "lepton_partons", "boost",
                                           "H_thad_tlep_ISR", "H_thad_tlep_ISR_cartesian"], dev=None, debug=False,
                 dtype=None, build=False, parton_list=[]):

        self.fields = {
            "partons": ["pt", "eta", "phi", "mass", "pdgId", "prov"],
            "boost": ["t", "x", "y", "z"],
            "incoming_particles_boost": ["t", "x", "y", "z"],
            "lepton_partons": ["pt", "eta", "phi", "mass", "pdgId"],
            "H_thad_tlep_ISR": ["pt", "eta", "phi", "mass"],
            "scaled_partons": ["pt", "eta", "phi", "prov"],
            "scaled_leptons": ["pt", "eta", "phi", "pdgId"],
            "H_thad_tlep_ISR_cartesian": ["E", "px", "py", "pz"]
        }

        tensors_bydefault = ['mask_partons', 'data_partons', 'mask_lepton_partons', 'data_lepton_partons',
                            'mask_boost', 'data_boost', 'data_higgs_t_tbar_ISR', 'data_higgs_t_tbar_ISR_cartesian',
                            'data_higgs_t_tbar_ISR_cartesian_onShell']
        
        
        print("PartonLevel CM")
        self.debug = debug

        self.root = root
        if root.endswith(".parquet"):
            self.rootDir = root.replace(".parquet","")
        else:
            self.rootDir = root

        if not os.path.exists(self.rootDir):
            build=True

        self.parton_list = parton_list
        os.makedirs(self.rootDir + "/processed_partons", exist_ok=True)
        self.object_types = object_types
        self.build = build

        # if build flag set or number of files in processed partons directory is 0
        if (build or len(os.listdir(self.rootDir + '/processed_partons/')) == 0):

            (self.partons_boosted, self.leptons_boosted, self.generator,
            self.gluon_ISR, self.boost) = self.get_PartonsAndLeptonsAndBoost()

            self.process('partons')
            self.process('boost')
            self.process('lepton_partons')

            self.process_intermediateParticles()
            self.process_intermediateParticles_cartesian()

            #for object_type in self.object_types:
            #    print("Create new file for " + object_type)
            #    if object_type == "H_thad_tlep_ISR":
            #        self.process_intermediateParticles()
            #    elif object_type == "H_thad_tlep_ISR_cartesian":
            #        self.process_intermediateParticles_cartesian()
            #    else:
            #        self.process(object_type)
            
            # need these files for the next operations
            self.mask_boost, self.data_boost = torch.load(
                                    self.processed_file_names("boost"))
            self.data_higgs_t_tbar_ISR_cartesian = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR_cartesian"))
            self.data_higgs_t_tbar_ISR = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR"))

            print("Create flattening weight")
            self.get_weight_flatetas()

            print("Create new file for data_higgs_t_tbar_ISR_cartesian_onShell")
            self.get_intermediateParticles_cartesian_onShell()

            self.data_higgs_t_tbar_ISR_cartesian_onShell = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR_cartesian_onShell"))

            print("Create new file for PhaseSpace + rambo detJacobian")
            self.get_PS_intermediateParticles()

            print("Create new file for PhaseSpace_onShell (no negative values)")
            self.get_PS_intermediateParticles_onShell()

            print("Create new file for Log_H_thad_tlep_ISR_cartesian")
            self.ProcessCartesianScaled()

            self.mean_log_data_higgs_t_tbar_ISR_cartesian, self.std_log_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR_cartesian"))
            self.logScaled_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_cartesian"))

            print("Create new file for Log_H_thad_tlep_ISR")
            self.ProcessScaled()

            print("Create new file for ScaledDecayProducts")
            self.ProcessScaledDecayProducts()

            

        print("Reading parton_level Files")

        self.mask_partons, self.data_partons = torch.load(
            self.processed_file_names("partons"))
        self.mask_lepton_partons, self.data_lepton_partons = torch.load(
            self.processed_file_names("lepton_partons"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))
        self.data_higgs_t_tbar_ISR = torch.load(
            self.processed_file_names("H_thad_tlep_ISR"))

        if "flattening_weight_HEta_tHadEta_tLepEta" in self.parton_list:
            self.flattening_weight_HEta_tHadEta_tLepEta = torch.load(
                self.processed_file_names("flattening_weight_HEta_tHadEta_tLepEta")
            )

        self.data_higgs_t_tbar_ISR_cartesian = torch.load(
            self.processed_file_names("H_thad_tlep_ISR_cartesian"))
        self.data_higgs_t_tbar_ISR_cartesian_onShell = torch.load(
            self.processed_file_names("H_thad_tlep_ISR_cartesian_onShell"))

        if 'phasespace_intermediateParticles' in self.parton_list:
            print("Load phasespace_intermediateParticles")
            self.phasespace_intermediateParticles = torch.load(
                self.processed_file_names("phasespace_intermediateParticles"))

        if 'phasespace_intermediateParticles_onShell' in self.parton_list:
            print("Load phasespace_intermediateParticles_onShell")
            self.phasespace_intermediateParticles_onShell = torch.load(
                self.processed_file_names("phasespace_intermediateParticles_onShell"))
            self.phasespace_rambo_detjacobian_onShell = torch.load(
                self.processed_file_names("phasespace_rambo_detjacobian_onShell"))

        if 'phasespace_intermediateParticles_onShell_logit' or 'phasespace_intermediateParticles_onShell_logit_scaled' in self.parton_list:
            print("Load phasespace_intermediateParticles_onShell_logit")
            self.phasespace_intermediateParticles_onShell_logit = torch.load(
                self.processed_file_names("phasespace_intermediateParticles_onShell_logit"))
            self.phasespace_intermediateParticles_onShell_logit_scaled = torch.load(
                self.processed_file_names("phasespace_intermediateParticles_onShell_logit_scaled"))
            self.mean_phasespace_intermediateParticles_onShell_logit, \
                self.std_phasespace_intermediateParticles_onShell_logit = torch.load(
                    self.processed_file_names("mean_std_phasespace_intermediateParticles_onShell_logit"))
            self.phasespace_rambo_detjacobian_onShell = torch.load(
                self.processed_file_names("phasespace_rambo_detjacobian_onShell"))
            

        if 'phasespace_rambo_detjacobian' in self.parton_list:
            print("Load phasespace_rambo_detjacobian")
            self.phasespace_rambo_detjacobian = torch.load(
                self.processed_file_names("phasespace_rambo_detjacobian"))
    
        if 'logScaled_data_higgs_t_tbar_ISR_cartesian' in self.parton_list or 'mean_log_data_higgs_t_tbar_ISR_cartesian' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR_cartesian")
            self.mean_log_data_higgs_t_tbar_ISR_cartesian, self.std_log_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR_cartesian"))
            self.logScaled_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_cartesian"))

        if 'logScaled_data_higgs_t_tbar_ISR' in self.parton_list or 'mean_log_data_higgs_t_tbar_ISR' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR")
            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
            self.logScaled_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR"))

        if 'logScaled_data_higgs_t_tbar_ISR_withEnergy' in self.parton_list or 'mean_log_data_higgs_t_tbar_ISR_withEnergy' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR_withEnergy")
            self.mean_log_data_higgs_t_tbar_ISR_withEnergy, self.std_log_data_higgs_t_tbar_ISR_withEnergy = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR_withEnergy"))
            self.logScaled_data_higgs_t_tbar_ISR_withEnergy = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_withEnergy"))

        if 'LogScaled_partonsLeptons' in self.parton_list or 'Log_mean_std_partonsLeptons' in self.parton_list:
            print("Load LogScaled_partonsLeptons")
            self.mean_log_partonsLeptons, self.std_log_partonsLeptons = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons"))
            self.LogScaled_partonsLeptons = torch.load(
                self.processed_file_names("LogScaled_partonsLeptons"))
            self.mask_partonsLeptons = torch.load(
                self.processed_file_names("mask_partonsLeptons"))

        if dtype != None:            
           for field in self.parton_list:
                setattr(self, field, getattr(self, field).to(dtype))
           for field in tensors_bydefault:
               setattr(self, field, getattr(self, field).to(dtype))

        print(f"Parton: Move tensors to device ({dev}) memory")
        for field in self.parton_list:
            setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
        for field in tensors_bydefault:
            setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
            
    @property
    def raw_file_names(self):
        return [self.root]

    def processed_file_names(self, type):
        return (self.rootDir + '/processed_partons/' + type + '_data.pt')

    def get_PartonsAndLeptonsAndBoost(self):
        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            generator = df["generator_info"]

            incoming_particles_boost = self.get_incoming_particles_boost(
                generator)

            boost = incoming_particles_boost

            partons = df["partons"]
            partons = ak.with_name(partons, name="Momentum4D")

            gluon = partons[partons.prov == 4]
            gluon = self.Reshape(gluon, utils.struct_gluon, 1)[:, 0]

            leptons = df["lepton_partons"]
            leptons = ak.with_name(leptons, name="Momentum4D")

            partons_boosted = self.boost_CM(partons, boost)
            leptons_boosted = self.boost_CM(leptons, boost)
            #higgs_boosted = self.boost_CM(higgs, boost) REMOVE the higgs --> use the sum of the partons
            gluon_ISR_boosted = self.boost_CM(gluon, boost)

        return partons_boosted, leptons_boosted, generator, gluon_ISR_boosted, boost

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
        # objects_array["rho"] = objects_CM.pt
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

            elif (object_type in ["boost"]):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))

            tensor_data = torch.tensor(d_list)
            tensor_mask = torch.tensor(mask)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))

    def process_intermediateParticles(self):
        higgs = self.get_Higgs()
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()
        gluon_ISR = self.gluon_ISR

        # residual boost
        boost_residual = higgs + top_hadronic + top_leptonic + gluon_ISR

        higgs = higgs.boost_p4(boost_residual.neg3D)
        top_hadronic = top_hadronic.boost_p4(boost_residual.neg3D)
        top_leptonic = top_leptonic.boost_p4(boost_residual.neg3D)
        gluon_ISR = gluon_ISR.boost_p4(boost_residual.neg3D)

        # Don't need to boost in CM frame because the partons/leptons are already boosted

        intermediate = [higgs, top_hadronic, top_leptonic, gluon_ISR]

        for i, objects in enumerate(intermediate):
            d_list = utils.to_flat_numpy(
                objects, self.fields["H_thad_tlep_ISR"], axis=1, allow_missing=False)

            d_list = np.expand_dims(d_list, axis=1)

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_data = torch.tensor(intermediate_np)
        torch.save(tensor_data, self.processed_file_names("H_thad_tlep_ISR"))

    def process_intermediateParticles_cartesian(self):
        higgs = self.get_Higgs()
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()
        gluon_ISR = self.gluon_ISR

        boost_residual = higgs + top_hadronic + top_leptonic + gluon_ISR

        higgs = higgs.boost_p4(boost_residual.neg3D)
        top_hadronic = top_hadronic.boost_p4(boost_residual.neg3D)
        top_leptonic = top_leptonic.boost_p4(boost_residual.neg3D)
        gluon_ISR = gluon_ISR.boost_p4(boost_residual.neg3D)

        # Don't need to boost in CM frame because the partons/leptons are already boosted
        intermediate = [higgs, top_hadronic, top_leptonic, gluon_ISR]

        for i, objects in enumerate(intermediate):

            objects_cartesian = self.change_to_cartesianCoordinates(objects)

            d_list = utils.to_flat_numpy(
                objects_cartesian, self.fields["H_thad_tlep_ISR_cartesian"], axis=1, allow_missing=False)

            d_list = np.expand_dims(d_list, axis=1)

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_data = torch.tensor(intermediate_np)
        torch.save(tensor_data, self.processed_file_names(
            "H_thad_tlep_ISR_cartesian"))


    def get_intermediateParticles_cartesian_onShell(self):
        tensor_cartesian_onShell = self.data_higgs_t_tbar_ISR_cartesian.clone()
        mass = torch.Tensor([M_HIGGS, M_TOP, M_TOP, M_GLUON])
        for i in range(4):
            tensor_cartesian_onShell[:,i,0] = torch.sqrt(tensor_cartesian_onShell[:,i,1]**2 + tensor_cartesian_onShell[:,i,2]**2 + tensor_cartesian_onShell[:,i,3]**2 + mass[i]**2)

        torch.save(tensor_cartesian_onShell, self.processed_file_names(
            "H_thad_tlep_ISR_cartesian_onShell"))


    def get_PS_intermediateParticles_onShell(self):
        E_CM = 13000
        mass = torch.Tensor([M_HIGGS, M_TOP, M_TOP, M_GLUON])
        phasespace = PhaseSpace(E_CM, [21, 21], [25, 6, -6, 21], final_masses=mass, dev="cpu")

        incoming_p_boost = self.data_boost

        # Check for miniaml energy requirement. There is one event out of 2M which <1GeV difference
        incoming_p_boost[:, 0, 0]  = torch.where(torch.sqrt(incoming_p_boost[:, 0, 0]**2 - incoming_p_boost[:, 0, 3]**2) < mass.sum(),
                                     torch.sqrt(incoming_p_boost[:, 0, 3]**2 + mass.sum()**2 + 1e-3),
                                     incoming_p_boost[:, 0, 0])
 
        x1 = torch.clamp((incoming_p_boost[:, 0, 0] + incoming_p_boost[:, 0, 3]) / E_CM, min=0., max=1.)
        x2 = torch.clamp((incoming_p_boost[:, 0, 0] - incoming_p_boost[:, 0, 3]) / E_CM, min=0., max=1.)
            
        ps, detjinv = phasespace.get_ps_from_momenta(
            self.data_higgs_t_tbar_ISR_cartesian_onShell, x1, x2)

        logit_ps = torch.logit(ps, eps=5e-5)
        ps_mean = logit_ps.nanmean(0)
        ps_scale = torch.sqrt(torch.nanmean(torch.pow(logit_ps, 2), 0) - torch.pow(ps_mean,2)) * 5 # to stay in -1,1 range
        logit_ps_scaled = (logit_ps - ps_mean) / ps_scale
    
        torch.save(ps, self.processed_file_names(
            "phasespace_intermediateParticles_onShell"))
        
        torch.save(logit_ps, self.processed_file_names(
            "phasespace_intermediateParticles_onShell_logit"))

        torch.save(logit_ps_scaled, self.processed_file_names(
            "phasespace_intermediateParticles_onShell_logit_scaled"))

        torch.save((ps_mean,ps_scale), self.processed_file_names(
            "mean_std_phasespace_intermediateParticles_onShell_logit"
        ))
        
        torch.save(detjinv, self.processed_file_names("phasespace_rambo_detjacobian_onShell"))

    def get_PS_intermediateParticles(self):

        E_CM = 13000
        mass = torch.Tensor([M_HIGGS, M_TOP, M_TOP, M_GLUON])
        phasespace = PhaseSpace(E_CM, [21, 21], [25, 6, -6, 21], final_masses=mass, dev="cpu")

        incoming_p_boost = self.data_boost
        x1 = (incoming_p_boost[:, 0, 0] + incoming_p_boost[:, 0, 3]) / E_CM
        x2 = (incoming_p_boost[:, 0, 0] - incoming_p_boost[:, 0, 3]) / E_CM
        ps, detjinv = phasespace.get_ps_from_momenta(
            self.data_higgs_t_tbar_ISR_cartesian, x1, x2)
    
        torch.save(ps, self.processed_file_names(
            "phasespace_intermediateParticles"))
        torch.save(detjinv, self.processed_file_names("phasespace_rambo_detjacobian"))
        
        
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
    
    def ProcessCartesianScaled(self):
        intermediateParticles = self.data_higgs_t_tbar_ISR_cartesian        
        log_intermediateParticles = torch.sign(intermediateParticles)*torch.log(1+torch.abs(intermediateParticles))
        
        mean_LogIntermediateParticles = torch.mean(log_intermediateParticles, dim=(0,1))
        std_LogIntermediateParticles = torch.std(log_intermediateParticles, dim=(0,1))
        
        scaledIntermediateParticles = \
            (log_intermediateParticles - mean_LogIntermediateParticles[None,None,:])/std_LogIntermediateParticles[None,None,:]
        
        #torch.save(log_intermediateParticles, self.processed_file_names(
        #    "Log_H_thad_tlep_ISR_cartesian"))
        torch.save((mean_LogIntermediateParticles, std_LogIntermediateParticles), self.processed_file_names(
            "Log_mean_std_H_thad_tlep_ISR_cartesian"))
        torch.save(scaledIntermediateParticles, self.processed_file_names(
            "LogScaled_H_thad_tlep_ISR_cartesian"))

    def ProcessScaled(self):
        intermediateParticles = self.data_higgs_t_tbar_ISR
        log_intermediateParticles = intermediateParticles[:,:,:3]

        pt = intermediateParticles[:,:,0]
        log_pt = torch.log(1+pt)
        log_intermediateParticles[:,:,0] = log_pt
        
        mean_LogIntermediateParticles = torch.mean(log_intermediateParticles, dim=(0,1))
        std_LogIntermediateParticles = torch.std(log_intermediateParticles, dim=(0,1))
        
        scaledIntermediateParticles = \
            (log_intermediateParticles - mean_LogIntermediateParticles[None,None,:])/std_LogIntermediateParticles[None,None,:]
        
        #torch.save(log_intermediateParticles, self.processed_file_names(
        #    "Log_H_thad_tlep_ISR"))
        torch.save((mean_LogIntermediateParticles, std_LogIntermediateParticles), self.processed_file_names(
            "Log_mean_std_H_thad_tlep_ISR"))
        torch.save(scaledIntermediateParticles, self.processed_file_names(
            "LogScaled_H_thad_tlep_ISR"))

        # Scaling also the boost
        boost_output = torch.cat((torch.from_numpy(self.boost.E.to_numpy()).unsqueeze(1),
                                  torch.from_numpy(self.boost.pz.to_numpy()).unsqueeze(1)), 1)
        boost_output[:,0] = torch.log(1+boost_output[:,0])
        mean_boost = torch.mean(boost_output, dim=0)
        std_boost = torch.std(boost_output, dim=0)
        scaled_boost_output = (boost_output - mean_boost)/std_boost
        torch.save((mean_boost, std_boost), self.processed_file_names("Log_mean_std_boost"))
        torch.save(scaled_boost_output, self.processed_file_names("LogScaled_boost"))

        # add energy in [pt, eta, phi]
        energy = self.logScaled_data_higgs_t_tbar_ISR_cartesian[:,:,0]
        energy_mean = self.mean_log_data_higgs_t_tbar_ISR_cartesian[0]
        energy_std = self.std_log_data_higgs_t_tbar_ISR_cartesian[0]

        scaledIntermediateParticles_withEnergy = torch.cat((energy.unsqueeze(dim=2), scaledIntermediateParticles), dim=2)
        mean_LogIntermediateParticles_withEnergy = torch.cat((energy_mean.reshape(1), mean_LogIntermediateParticles), dim=0)
        std_LogIntermediateParticles_withEnergy = torch.cat((energy_std.reshape(1), std_LogIntermediateParticles), dim=0)

        torch.save((mean_LogIntermediateParticles_withEnergy, std_LogIntermediateParticles_withEnergy), self.processed_file_names(
            "Log_mean_std_H_thad_tlep_ISR_withEnergy"))
        torch.save(scaledIntermediateParticles_withEnergy, self.processed_file_names(
            "LogScaled_H_thad_tlep_ISR_withEnergy"))

    def ProcessScaledDecayProducts(self):
        
        # add here b jets from Higgs, thad, tlep
        # order: # H, H1, H2, thad, thad1, tlep, tlep1, ISR
        partons = self.partons_boosted

        # first position always massive leptons, 2nd position = neutrino
        leptons = self.leptons_boosted

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]
        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]
        # find partons with provenance 4 (ISR)
        # sometimes there are events without ISR -> paddinng = partons with prov=-1 which are null
        prov4_partons = partons[partons.prov == 4]
        # find partons with provenance 5 (light q)
        prov5_partons = partons[partons.prov == 5]
        # find partons with provenance -1 (not matched) -> I don't need to add this in data
        # because I will mask the missing partons (which are the ones with ISR)

        partons = [prov1_partons, prov2_partons, prov3_partons, prov5_partons, prov4_partons, leptons]

        for i, objects in enumerate(partons):

            if i == 5:
                # lepton case -> we don't have prov so we attach the pdgID
                d_list = utils.to_flat_numpy(
                    objects, self.fields["scaled_leptons"], axis=1, allow_missing=False)
            else:
                # case prov = 4 => ISR (sometimes missing)
                if (i == 4):
                    objects = self.Reshape(objects, utils.struct_gluon, 1)

                d_list = utils.to_flat_numpy(
                    objects, self.fields["scaled_partons"], axis=1, allow_missing=False)
    
            d_list = np.transpose(d_list, (0, 2, 1))

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_partonsLeptons = torch.tensor(intermediate_np) # H1 H2 thad1 thad2
        tensor_partonsLeptons_boxcox = torch.clone(tensor_partonsLeptons)
        mask_partonsLeptons = (tensor_partonsLeptons[...,3] != -1).bool() # pt > 0
        tensor_partonsLeptons[:,:,0] = torch.log(1 + tensor_partonsLeptons[:,:,0]) # take log of masked
        # I don't care because I will mask these elems when I do any computations

        mean_list = []
        std_list = []
        for i in range(3):
            feature = tensor_partonsLeptons[...,i]
            mean_feature = torch.mean(feature[mask_partonsLeptons]) # masked mean
            std_feature = torch.std(feature[mask_partonsLeptons]) # masked std
            mean_list.append(mean_feature)
            std_list.append(std_feature)

        mean_tensor_partonsLeptons = torch.Tensor(mean_list)
        std_tensor_partonsLeptons = torch.Tensor(std_list)

        parton_type = [1]* 7
        parton_type = [*parton_type, 2, 2] # 1 for jets, 2 for leptons
        parton_type = torch.Tensor(parton_type).unsqueeze(dim=1)
        parton_type = parton_type.unsqueeze(dim=0).repeat(tensor_partonsLeptons.size(0),1,1)

        tensor_partonsLeptons = torch.cat((tensor_partonsLeptons, parton_type), dim=2)
        
        tensor_partonsLeptons[:,:,:2] = \
            (tensor_partonsLeptons[:,:,:2] - mean_tensor_partonsLeptons[None,None,:2])/std_tensor_partonsLeptons[None,None,:2]

        # sort higgs and thad decay products (based on pt)
        tensor_partonsLeptons = self.sort_pt(tensor_partonsLeptons)
        # I don't need to modify the mask cuz ISR is in the same position
        
        torch.save((mean_tensor_partonsLeptons, std_tensor_partonsLeptons), self.processed_file_names(
            "Log_mean_std_partonsLeptons"))
        torch.save(tensor_partonsLeptons, self.processed_file_names(
            "LogScaled_partonsLeptons"))
        torch.save(mask_partonsLeptons, self.processed_file_names(
            "mask_partonsLeptons"))


    def sort_pt(self, tensor_AllPartons):
        pt_0 = tensor_AllPartons[:,0,0]
        pt_1 = tensor_AllPartons[:,1,0]

        mask_higgs = pt_0 > pt_1

        # sort higgs based on pt
        tensor_AllPartons[:,[0,1]] = torch.where(mask_higgs[:,None,None],
                                                    tensor_AllPartons[:,[0,1]],
                                                    tensor_AllPartons[:,[1,0]])

        #pt_2 = tensor_AllPartons[:,2,0]
        #pt_4 = tensor_AllPartons[:,4,0]
        
        #mask_thad1 = pt_2 > pt_4        

        # sort thad based on pt
        #tensor_AllPartons[:,[2,4]] = torch.where(mask_thad1[:,None,None],
        #                                            tensor_AllPartons[:,[2,4]],
        #                                            tensor_AllPartons[:,[4,2]])

        pt_4 = tensor_AllPartons[:,4,0]
        pt_5 = tensor_AllPartons[:,5,0]
        
        mask_thad2 = pt_4 > pt_5 

        # sort thad based on pt
        tensor_AllPartons[:,[4,5]] = torch.where(mask_thad2[:,None,None],
                                                    tensor_AllPartons[:,[4,5]],
                                                    tensor_AllPartons[:,[5,4]])

        #pt_2 = tensor_AllPartons[:,2,0]
        #pt_4 = tensor_AllPartons[:,4,0]
        
        #mask_thad3 = pt_2 > pt_4        

        # sort thad based on pt
        #tensor_AllPartons[:,[2,4]] = torch.where(mask_thad3[:,None,None],
        #                                            tensor_AllPartons[:,[2,4]],
        #                                            tensor_AllPartons[:,[4,2]])

        return tensor_AllPartons
        
        
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

    def get_weight_flatetas(self):
        dataCorrect = self.data_higgs_t_tbar_ISR
        higgs = dataCorrect[:,0]
        t = dataCorrect[:,1]
        tbar = dataCorrect[:,2]
        ISR = dataCorrect[:,3]
        Nbins = 25

        bins_h = np.linspace(-4,4, Nbins)
        bins_t = np.linspace(-4,4, Nbins)
        bins_tbar = np.linspace(-4,4, Nbins)
        
        h = hist.Hist(
            hist.axis.Variable( bins_h, name="h"),
            hist.axis.Variable( bins_t, name="t"),
            hist.axis.Variable( bins_tbar, name="tbar"),
        )
        h.fill(higgs[:,1], t[:,1], tbar[:,1])

        w3d = np.where(
            h.values()>0.,
            (1/ h.values()) * len(higgs)/(Nbins**3), 
            1.)
        w3d[w3d>30] = 30
        
        xind = np.digitize(higgs[:,1],  bins_h,  right=False)-1
        yind = np.digitize(t[:,1], bins_t,       right=False)-1
        zind = np.digitize(tbar[:,1], bins_tbar, right=False)-1
        index = np.stack([xind, yind, zind], axis=1)

        w = get_weight(w3d, index, Nbins)
        torch.save(torch.from_numpy(w), self.processed_file_names(
            "flattening_weight_HEta_tHadEta_tLepEta"))

    def __getitem__(self, index):
                    

        if self.debug == True:
            return (self.mask_partons[index], self.data_partons[index],
                    self.mask_lepton_partons[index], self.data_lepton_partons[index],
                    self.mask_boost[index], self.data_boost[index],
                    self.data_higgs_t_tbar_ISR[index], self.data_higgs_t_tbar_ISR_cartesian[index],
                    self.phasespace_intermediateParticles[index],
                    self.phasespace_rambo_detjacobian[index],
                    self.log_data_higgs_t_tbar_ISR_cartesian[index],
                    # no index for mean/std because size is [4]
                    self.mean_log_data_higgs_t_tbar_ISR_cartesian, self.std_log_data_higgs_t_tbar_ISR_cartesian,
                    self.logScaled_data_higgs_t_tbar_ISR_cartesian[index])


        out = [ ]
        for field in self.parton_list:
             if field not in ['mean_log_data_higgs_t_tbar_ISR_cartesian',
                              'std_log_data_higgs_t_tbar_ISR_cartesian',
                              'mean_log_data_higgs_t_tbar_ISR',
                              'std_log_data_higgs_t_tbar_ISR',
                              'mean_phasespace_intermediateParticles_onShell_logit',
                              'std_phasespace_intermediateParticles_onShell_logit',
                              'Log_mean_std_partonsLeptons']:
                 out.append(getattr(self, field)[index])
        return out

    def __len__(self):
        size = len(self.mask_partons)
        return size
