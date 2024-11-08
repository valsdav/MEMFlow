from memflow.read_data import utils
import os
import os.path
import torch
import numpy as np
import awkward as ak
from torch.utils.data import Dataset
from scipy import stats

from .utils_orderJetsSpanet import higgsAssignment_SPANET
from .utils_orderJetsSpanet import sortObjects_bySpanet
from .utils_orderJetsSpanet import sortObjects_byProv


class Dataset_RecoLevel_NoBoost(Dataset):
    def __init__(self, root, object_types=["jets", "lepton_reco", "met", "boost"], dev=None, debug=False,
                 dtype=None, build=False, reco_list=[]):

        self.fields = {
            "jets": ["pt", "eta", "phi", "btag", "prov_Thad", "prov_Tlep", "prov_H", "prov"],
            "lepton_reco": ["pt", "eta", "phi"],
            "met": ["pt", "eta", "phi"],
            "boost": ["E", "px", "py", "pz"],
            "recoParticles_Cartesian": ["E", "px", "py", "pz"]
        }

        print("RecoLevel LAB")
        self.debug = debug
        
        self.root = root
        if root.endswith(".parquet"):
            self.rootDir = root.replace(".parquet","")
        else:
            self.rootDir = root

        if not os.path.exists(self.rootDir):
            build=True
            
        self.reco_list = reco_list

        os.makedirs(self.rootDir + "/processed_jets_noBoost", exist_ok=True)
        self.object_types = object_types

        allObjects = self.object_types[:]
        allObjects.append('recoParticles_Cartesian')
        # if build flag set or number of files in processed jets directory is 0
        if (build or  len(os.listdir(self.rootDir + '/processed_jets_noBoost/')) == 0):
            self.boost = self.get_boost()

            for object_type in self.object_types:
                print("Create new file for " + object_type)
                self.process(object_type)

            print("Create new file for recoParticles_Cartesian")
            self.processCartesian()

            # Need these tensors for scaleObjects
            self.mask_jets, self.data_jets = torch.load(self.processed_file_names("jets"))
            self.mask_lepton, self.data_lepton = torch.load(self.processed_file_names("lepton_reco"))
            self.mask_met, self.data_met = torch.load(self.processed_file_names("met"))
            self.mask_boost, self.data_boost = torch.load(self.processed_file_names("boost"))
            self.recoParticlesCartesian = torch.load(self.processed_file_names("recoParticles_Cartesian"))

            print("Create new file for LogData")
            self.scaleObjects()
            self.scaledLogRecoParticles, self.LogRecoParticles, self.meanRecoParticles, self.stdRecoParticles = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles'))

            self.scaledLogRecoParticles_fullCartesian, self.meanRecoParticles, self.stdRecoParticles, \
            self.meanRecoParticles_cartesian, self.stdRecoParticles_cartesian = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles_fullCartesian'))
            
            print("Create new file for sortJets_bySpanet")
            self.sortJets_bySpanet()

            print("Create new file for sortJets_bySpanet_ptcut")
            self.sortJets_bySpanet_ptcut()

            print("Create new file for sortJets_bySpanet_boxcox")
            self.sortJets_bySpanet_boxcox()

            print("Create new file for sortJets_byProv")
            self.sortJets_byMatched()

        

        self.mask_jets, self.data_jets = torch.load(self.processed_file_names("jets"))
        self.mask_lepton, self.data_lepton = torch.load(self.processed_file_names("lepton_reco"))
        self.mask_met, self.data_met = torch.load(self.processed_file_names("met"))
        self.mask_boost, self.data_boost = torch.load(self.processed_file_names("boost"))
        tensors_bydefault = ['mask_jets', 'data_jets', 'mask_lepton', 'data_lepton',
                            'mask_met', 'data_met', 'mask_boost', 'data_boost']
        
        print("Reading reco_level Files")
        if 'recoParticlesCartesian' in self.reco_list:
            print("Load recoParticles_Cartesian")
            self.recoParticlesCartesian = torch.load(self.processed_file_names("recoParticles_Cartesian"))
        
        if 'recoParticles' in self.reco_list:
            print("Load recoParticles")
            self.recoParticles = torch.load(self.processed_file_names('recoParticles'))
        
        if 'scaledLogJets' in self.reco_list:
            print("Load scaledLogJets")
            self.scaledLogJets, self.LogJets, self.meanJets, self.stdJets = torch.load(
                                                                self.processed_file_names('scaledLogJets'))

        if 'scaledLogLepton' in self.reco_list:
            print("Load scaledLogLepton")
            self.scaledLogLepton, self.LogLepton, self.meanLepton, self.stdLepton = \
                                            torch.load(self.processed_file_names('scaledLogLepton'))
        
        if 'scaledLogMet' in self.reco_list:
            print("Load scaledLogMet")
            self.scaledLogMet, self.LogMet, self.meanMet, self.stdMet = \
                                            torch.load(self.processed_file_names('scaledLogMet'))
        
        if 'scaledLogBoost' in self.reco_list:
            print("Load scaledLogBoost")
            self.scaledLogBoost, self.LogBoost, self.meanBoost, self.stdBoost = \
                                            torch.load(self.processed_file_names('scaledLogBoost'))
        
        if 'scaledLogRecoParticlesCartesian' in self.reco_list:
            print("Load scaledLogRecoParticlesCartesian")
            self.scaledLogRecoParticlesCartesian, self.LogRecoParticlesCartesian, self.meanRecoCartesian, self.stdRecoCartesian = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticlesCartesian'))

        if 'scaledLogRecoParticles' in self.reco_list:
            print("Load scaledLogRecoParticles")
            self.scaledLogRecoParticles, self.LogRecoParticles, self.meanRecoParticles, self.stdRecoParticles = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles'))

        if 'scaledLogRecoParticles_fullCartesian' in self.reco_list:
            print("Load scaledLogRecoParticles_fullCartesian")            
            self.scaledLogRecoParticles_fullCartesian, self.meanRecoParticles, self.stdRecoParticles, \
                self.meanRecoParticles_cartesian, self.stdRecoParticles_cartesian = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles_fullCartesian'))

        if 'scaledLogRecoParticles_withEnergy' in self.reco_list:
            print("Load scaledLogRecoParticles_withEnergy")
            self.scaledLogRecoParticles_withEnergy, self.meanRecoParticles_withEnergy, self.stdRecoParticles_withEnergy = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles_withEnergy'))
            
        if 'scaledLogReco_sortedBySpanet' in self.reco_list:
            print("Load scaledLogReco_sortedBySpanet")
            self.scaledLogRecoParticles_fullCartesian, self.meanRecoParticles, self.stdRecoParticles, \
                self.meanRecoParticles_cartesian, self.stdRecoParticles_cartesian = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles_fullCartesian'))
            
            self.scaledLogReco_sortedBySpanet, self.mask_scaledLogReco_sortedBySpanet, self.meanRecoParticles, self.stdRecoParticles = \
                                    torch.load(self.processed_file_names('scaledLogReco_sortedBySpanet'))
            self.mean_btagLogit, self.std_btagLogit = torch.load(self.processed_file_names('mean_btag_logit'))

        if 'scaledLogReco_sortedBySpanet_phiScaled' in self.reco_list:
            print("Load scaledLogReco_sortedBySpanet")
            self.scaledLogReco_sortedBySpanet_phiScaled, self.mask_scaledLogReco_sortedBySpanet, self.meanRecoParticles, self.stdRecoParticles = \
                                    torch.load(self.processed_file_names('scaledLogReco_sortedBySpanet_phiScaled'))
            self.mean_btagLogit, self.std_btagLogit = torch.load(self.processed_file_names('mean_btag_logit'))

        if 'scaledLogReco_sortedBySpanet_ptcut' in self.reco_list:
            print("Load scaledLogReco_sortedBySpanet_ptcut")
            self.scaledLogReco_sortedBySpanet_ptcut, self.mask_scaledLogReco_sortedBySpanet_ptcut, self.meanRecoParticles_ptcut, self.stdRecoParticles_ptcut, self.min_pt_eachReco = \
                                    torch.load(self.processed_file_names('scaledLogReco_sortedBySpanet_ptcut'))

        if 'scaledLogReco_sortedBySpanet_boxcox' in self.reco_list:
            print("Load scaledLogReco_sortedBySpanet_boxcox")
            self.scaledLogReco_sortedBySpanet_boxcox, self.mask_scaledLogReco_sortedBySpanet_boxcox, self.meanRecoParticles_boxcox, self.stdRecoParticles_boxcox = \
                                    torch.load(self.processed_file_names('scaledLogReco_sortedBySpanet_boxcox'))
            self.mean_btagLogit, self.std_btagLogit = torch.load(self.processed_file_names('mean_btag_logit'))
            self.lambda_boxcox = torch.load(self.processed_file_names('lambda_boxcox'))

        if 'scaledLogReco_sortedByProv' in self.reco_list:
            print("Load scaledLogReco_sortedByProv")
            self.scaledLogReco_sortedByProv, self.mask_scaledLogReco_sortedByProv, self.meanRecoParticles, self.stdRecoParticles = \
                                    torch.load(self.processed_file_names('scaledLogReco_sortedByProv'))

        if dtype != None:
            for field in self.reco_list:
                setattr(self, field, getattr(self, field).to(dtype))
            for field in tensors_bydefault:
                setattr(self, field, getattr(self, field).to(dtype))

        print(f"Reco: Move tensors to device ({dev}) memory")
        for field in self.reco_list:
            setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
        for field in tensors_bydefault:
            setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
            
            

    @property
    def raw_file_names(self):
        return [self.root]

    def processed_file_names(self, type):

        return (self.rootDir + '/processed_jets_noBoost/' + type + '_data.pt')

    def get_boost(self):

        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            jets = df["jets"]
            jets = ak.with_name(jets, name="Momentum4D")

            leptons = df["lepton_reco"]
            leptons = ak.with_name(leptons, name="Momentum4D")[:,0] # taking the leading lepton

            met = df["met"]
            met = ak.with_name(met, name="Momentum4D")

            boost_jets = utils.get_vector_sum(jets)
            boost = boost_jets + leptons + met

        return boost

    def Reshape(self, input, value, ax, max_no=None):
        if max_no is None:
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

            if object_type == "jets":
                objects = self.Reshape(objects, utils.struct_jets, 1, max_no=16)

            if object_type == "lepton_reco":
                # taking the leading lepton
                objects = objects[:,0]
            
            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "jets":
                d_list = np.transpose(d_list, (0, 2, 1))
                mask = self.get_mask_pt(d_list)

            if (object_type == "lepton_reco" or object_type == "met" or object_type == "boost"):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))
            
            if (object_type == "lepton_reco" or object_type == "met"):
                #missing fields
                missing_fields = np.zeros((d_list.shape[0], d_list.shape[1],
                                           len(self.fields["jets"])-len(self.fields[object_type])
                                           ))
                d_list = np.concatenate((d_list, missing_fields), axis=2)
                
            tensor_data = torch.tensor(d_list, dtype=torch.float)
            tensor_mask = torch.tensor(mask, dtype=torch.float)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))
            
    def processCartesian(self):
        
        objects = ["jets", "lepton_reco", "met"] # don't need boost
        
        # Don't need mask -> concat jet/lepton/met masks
        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            for object_type in objects:
                
                objects = ak.with_name(df[object_type], name="Momentum4D")

                if object_type == "jets":
                    objects = self.Reshape(objects, utils.struct_jets, 1, 16)

                if object_type == "lepton_reco":
                    # only the leading lepton
                    objects = objects[:,0]
                    
                d_list = utils.to_flat_numpy(
                        objects, self.fields['recoParticles_Cartesian'], axis=1, allow_missing=False)

                if object_type == "jets":
                    d_list = np.transpose(d_list, (0, 2, 1))

                if (object_type == "lepton_reco" or object_type == "met"):
                    d_list = np.expand_dims(d_list, axis=1)

                tensor_data = torch.tensor(d_list, dtype=torch.float)

                if object_type == "jets":
                    recoParticlesCartesian = tensor_data
                else:
                    recoParticlesCartesian = torch.cat((recoParticlesCartesian, tensor_data), 1)
            
            
            torch.save((recoParticlesCartesian),
                           self.processed_file_names('recoParticles_Cartesian'))
    
    
    def scaleTensor(self, object_tensor, mask, isCartesian=False):
        
        # get masks of 'recoParticlesCartesian'
        
        maskNumpy = mask.numpy()
        log_objectTensor = object_tensor.clone()

        # pt or E.  all the rest we don't log scale it
        log_objectTensor[:,:,0] = torch.sign(log_objectTensor[:,:,0])*torch.log(1+torch.abs(log_objectTensor[:,:,0]))
        if isCartesian:
            # case [E,px,py,pz]
            log_objectTensor[:,:,1:4] = torch.sign(log_objectTensor[:,:,1:4])*torch.log(1+torch.abs(log_objectTensor[:,:,1:4]))
            no_elements = 4             
        else:
            # case ["pt", "eta", "phi", "btag", + prov]
            no_elements = 3
            
        for i in range(no_elements):
            feature = log_objectTensor[:,:,i].numpy()
            feature_masked = np.ma.masked_array(feature, mask=np.logical_not(maskNumpy))

            feature_scaled = (feature_masked - feature_masked.mean())/feature_masked.std()
            feature_scaled[np.logical_not(maskNumpy)] = 0
            
            # use only the ndarray from masked array
            feature_scaled = np.ma.getdata(feature_scaled)
            feature_tensor = torch.tensor(feature_scaled, dtype=torch.float).unsqueeze(dim=2)
            mean_tensor = torch.tensor(feature_masked.mean(), dtype=torch.float).reshape(1)
            std_tensor = torch.tensor(feature_masked.std(), dtype=torch.float).reshape(1)
            if i == 0:
                logScaled_objectTensor = feature_tensor
                meanLogTensor = mean_tensor
                stdLogTensor = std_tensor
            else:
                logScaled_objectTensor = torch.cat((logScaled_objectTensor, feature_tensor), dim=2)
                meanLogTensor = torch.cat((meanLogTensor, mean_tensor), dim=0)
                stdLogTensor = torch.cat((stdLogTensor, std_tensor), dim=0)
        
        # overwrite in object_tensor the new logScaled_objectTensor (to keep btag and prov)
        # then reassign logScaled_objectTensor (because this is the object returned)
        if isCartesian == False:
            logScaled_objectTensor = torch.cat((logScaled_objectTensor, object_tensor[:,:,3:]), dim=2)
            
        return meanLogTensor, stdLogTensor, logScaled_objectTensor, log_objectTensor
                
        
    def scaleObjects(self):
        
        meanJets, stdJets, scaledLogJets, LogJets = self.scaleTensor(self.data_jets,
                                                                     self.mask_jets, isCartesian=False)
        meanLepton, stdLepton, scaledLogLepton, LogLepton = self.scaleTensor(self.data_lepton, self.mask_lepton,
                                                                isCartesian=False)
        meanMet, stdMet, scaledLogMet, LogMet = self.scaleTensor(self.data_met, self.mask_met, isCartesian=False)
        
        meanBoost, stdBoost, scaledLogBoost, LogBoost = self.scaleTensor(self.data_boost,
                                                                         self.mask_boost, isCartesian=True)
        recoMask = torch.cat((self.mask_jets, self.mask_lepton, self.mask_met), dim=1)
        meanRecoCartesian, stdRecoCartesian, scaledLogRecoParticlesCartesian, LogRecoParticlesCartesian = \
                                           self.scaleTensor(self.recoParticlesCartesian,
                                                            recoMask, isCartesian=True)
        
        recoParticles = torch.cat((self.data_jets, self.data_lepton, self.data_met), dim=1)
        meanRecoParticles, stdRecoParticles, scaledLogRecoParticles, LogRecoParticles = self.scaleTensor(recoParticles,
                                                                                  recoMask, isCartesian=False)

        # attach btag and prov to cartesian full tensor
        scaledLogRecoParticles_fullCartesian = torch.cat((scaledLogRecoParticles[:,:,:3],
                                                          scaledLogRecoParticlesCartesian,
                                                          recoParticles[:,:,3:]), dim=2)
        
        # attach btag and prov to cartesian full tensor
        scaledLogRecoParticlesCartesian = torch.cat((scaledLogRecoParticlesCartesian, recoParticles[:,:,3:]), dim=2)
        LogRecoParticlesCartesian = torch.cat((LogRecoParticlesCartesian, recoParticles[:,:,3:]), dim=2)
        
        
        torch.save((recoParticles), self.processed_file_names('recoParticles'))
        
        #torch.save((scaledLogJets, LogJets, meanJets, stdJets), self.processed_file_names('scaledLogJets'))
        #torch.save((scaledLogLepton, LogLepton, meanLepton, stdLepton), self.processed_file_names('scaledLogLepton'))
        #torch.save((scaledLogMet, LogMet, meanMet, stdMet), self.processed_file_names('scaledLogMet'))
        torch.save((scaledLogBoost, LogBoost, meanBoost, stdBoost), self.processed_file_names('scaledLogBoost'))
        torch.save((scaledLogRecoParticlesCartesian, LogRecoParticlesCartesian, meanRecoCartesian, stdRecoCartesian),
                       self.processed_file_names('scaledLogRecoParticlesCartesian'))
        torch.save((scaledLogRecoParticles, LogRecoParticles, meanRecoParticles, stdRecoParticles),
                       self.processed_file_names('scaledLogRecoParticles'))
        torch.save((scaledLogRecoParticles_fullCartesian, meanRecoParticles, stdRecoParticles, meanRecoCartesian, stdRecoCartesian),
                       self.processed_file_names('scaledLogRecoParticles_fullCartesian'))

        energy = scaledLogRecoParticlesCartesian[:,:,0]
        energy_mean = meanRecoCartesian[0]
        energy_std = stdRecoCartesian[0]

        #[E, pt, eta, phi]
        scaledLogRecoParticles_withEnergy = torch.cat((energy.unsqueeze(dim=2), scaledLogRecoParticles), dim=2)
        meanRecoParticles_withEnergy = torch.cat((energy_mean.reshape(1), meanRecoParticles), dim=0)
        stdRecoParticles_withEnergy = torch.cat((energy_std.reshape(1), stdRecoParticles), dim=0)
        torch.save((scaledLogRecoParticles_withEnergy, meanRecoParticles_withEnergy, stdRecoParticles_withEnergy),
                    self.processed_file_names('scaledLogRecoParticles_withEnergy'))
        
    def sortJets_bySpanet(self):
        spanet_tensor = self.scaledLogRecoParticles[:,:16,4:7] # only jets
        spanet_assignment = higgsAssignment_SPANET(spanet_values=spanet_tensor)
        
        # order H1, H2, thad1, thad2, thad3, tlep1, lepton, MET
        scaledLogReco_sortedBySpanet = sortObjects_bySpanet(spanet_assignment=spanet_assignment, scaledLogReco=self.scaledLogRecoParticles_fullCartesian,
                                               maskJets=self.mask_jets, order=[0, 1, 2])

        # at this point the missing Spanet jets and the padding have -100 values
        # 'scaledLogReco_sortedBySpanet' doesn't contain the 'exist' flag for each jet
        exist = torch.where((scaledLogReco_sortedBySpanet[:,:,0] != -100), 1, 0).unsqueeze(dim=2)
        # attach exist flag to each jet
        scaledLogReco_sortedBySpanet = torch.cat((exist, scaledLogReco_sortedBySpanet), dim=2)

        # check exist flag is ok
        if torch.count_nonzero((scaledLogReco_sortedBySpanet[...,0]*scaledLogReco_sortedBySpanet[...,1]) == -100) > 0:
            raise Exception("Check mask... this product must be 0")

        # last part: modify the missing SPANET jets to be [-1...]
        # keep padding jets as [-100...]
        # take the first 6 spanet jets
        scaledLogReco_sortedBySpanet[:,:6,:] = torch.where((scaledLogReco_sortedBySpanet[:,:6,:] == -100), -1, scaledLogReco_sortedBySpanet[:,:6,:])

        # check if the change works (8 because lepton and MET are not -100)
        if torch.count_nonzero(scaledLogReco_sortedBySpanet[:,:8,:] == -100) > 0:
            raise Exception("Missing jets are still -100")

        maskExist = scaledLogReco_sortedBySpanet[...,0] == 1

        # add a column with (torch.logit(btag) - mean)/std
        # save also mean and std
        # I don't care about nans values -> these will be masked during training
        btag_logit = torch.logit(scaledLogReco_sortedBySpanet[...,4], eps=0.002)
        mean_btag = torch.mean(btag_logit[maskExist], dim=0)
        std_btag = torch.std(btag_logit[maskExist], dim=0)
        btag_logit = (btag_logit - mean_btag)/std_btag
        btag_logit = btag_logit.unsqueeze(dim=2)
        scaledLogReco_sortedBySpanet = torch.cat((scaledLogReco_sortedBySpanet, btag_logit), dim=2)

        # unscale phi for periodic flows
        scaledLogReco_sortedBySpanet[...,3] = scaledLogReco_sortedBySpanet[...,3]*self.stdRecoParticles[...,2] + self.meanRecoParticles[...,2]

        # missing jets -> phi & btag = -1
        scaledLogReco_sortedBySpanet[:,:6,-1] = torch.where(~maskExist[:,:6], -1, scaledLogReco_sortedBySpanet[:,:6,-1])
        scaledLogReco_sortedBySpanet[:,:6,3] = torch.where(~maskExist[:,:6], -1, scaledLogReco_sortedBySpanet[:,:6,3])

        # padding jets as -100
        scaledLogReco_sortedBySpanet[:,6:,-1] = torch.where(~maskExist[:,6:], -100, scaledLogReco_sortedBySpanet[:,6:,-1])
        scaledLogReco_sortedBySpanet[:,6:,3] = torch.where(~maskExist[:,6:], -100, scaledLogReco_sortedBySpanet[:,6:,3])

        # scale phi back
        scaledLogReco_sortedBySpanet_phiScaled = torch.clone(scaledLogReco_sortedBySpanet)
        scaledLogReco_sortedBySpanet_phiScaled[...,3] = (scaledLogReco_sortedBySpanet_phiScaled[...,3] - self.meanRecoParticles[...,2])/self.stdRecoParticles[...,2]

        # check pt != -100
        mask_scaledLogReco_sortedBySpanet = (scaledLogReco_sortedBySpanet[:,:,1] != -100).bool()
        torch.save((scaledLogReco_sortedBySpanet, mask_scaledLogReco_sortedBySpanet, self.meanRecoParticles, self.stdRecoParticles),
                    self.processed_file_names('scaledLogReco_sortedBySpanet'))

        torch.save((mean_btag, std_btag), self.processed_file_names('mean_btag_logit'))

        torch.save((scaledLogReco_sortedBySpanet_phiScaled, mask_scaledLogReco_sortedBySpanet, self.meanRecoParticles, self.stdRecoParticles),
                    self.processed_file_names('scaledLogReco_sortedBySpanet_phiScaled'))

    def sortJets_bySpanet_ptcut(self):
        spanet_tensor = self.scaledLogRecoParticles[:,:16,4:7] # only jets
        spanet_assignment = higgsAssignment_SPANET(spanet_values=spanet_tensor)
        
        # order H1, H2, thad1, thad2, thad3, tlep1, lepton, MET
        scaledLogReco_sortedBySpanet = sortObjects_bySpanet(spanet_assignment=spanet_assignment, scaledLogReco=self.scaledLogRecoParticles_fullCartesian,
                                               maskJets=self.mask_jets, order=[0, 1, 2])

        # at this point the missing Spanet jets and the padding have -100 values
        # 'scaledLogReco_sortedBySpanet' doesn't contain the 'exist' flag for each jet
        exist = torch.where((scaledLogReco_sortedBySpanet[:,:,0] != -100), 1, 0).unsqueeze(dim=2)
        # attach exist flag to each jet
        scaledLogReco_sortedBySpanet = torch.cat((exist, scaledLogReco_sortedBySpanet), dim=2)

        # check exist flag is ok
        if torch.count_nonzero((scaledLogReco_sortedBySpanet[...,0]*scaledLogReco_sortedBySpanet[...,1]) == -100) > 0:
            raise Exception("Check mask... this product must be 0")

        # last part: modify the missing SPANET jets to be [-1...]
        # keep padding jets as [-100...]
        # take the first 6 spanet jets
        scaledLogReco_sortedBySpanet[:,:6,:] = torch.where((scaledLogReco_sortedBySpanet[:,:6,:] == -100), -1, scaledLogReco_sortedBySpanet[:,:6,:])

        # check if the change works (8 because lepton and MET are not -100)
        if torch.count_nonzero(scaledLogReco_sortedBySpanet[:,:8,:] == -100) > 0:
            raise Exception("Missing jets are still -100")

        maskExist = scaledLogReco_sortedBySpanet[...,0] == 1

        # add a column with (torch.logit(btag) - mean)/std
        # save also mean and std
        # I don't care about nans values -> these will be masked during training
        btag_logit = torch.logit(scaledLogReco_sortedBySpanet[...,4], eps=0.002)
        mean_btag = torch.mean(btag_logit[maskExist], dim=0)
        std_btag = torch.std(btag_logit[maskExist], dim=0)
        btag_logit = (btag_logit - mean_btag)/std_btag
        btag_logit = btag_logit.unsqueeze(dim=2)
        scaledLogReco_sortedBySpanet = torch.cat((scaledLogReco_sortedBySpanet, btag_logit), dim=2)

        # unscale phi for periodic flows + unscale pt for cut
        scaledLogReco_sortedBySpanet[...,[1,3]] = scaledLogReco_sortedBySpanet[...,[1,3]]*self.stdRecoParticles[...,[0,2]] + self.meanRecoParticles[...,[0,2]]
        # unscale pt
        scaledLogReco_sortedBySpanet[...,1] = torch.exp(scaledLogReco_sortedBySpanet[...,1]) - 1

        min_pt_eachReco = torch.empty(0)

        # substract minimum pt for every reco object (only exist object)
        for i in range(scaledLogReco_sortedBySpanet.shape[1]):
            maskReco = scaledLogReco_sortedBySpanet[:,i,0] == 1
            
            if torch.count_nonzero(maskReco) > 0:
                print(torch.min(scaledLogReco_sortedBySpanet[:,i,1][maskReco]))
                min_pt_eachReco = torch.cat((min_pt_eachReco, torch.Tensor([torch.min(scaledLogReco_sortedBySpanet[:,i,1][maskReco])])))
                scaledLogReco_sortedBySpanet[:,i,1][maskReco] = scaledLogReco_sortedBySpanet[:,i,1][maskReco] - torch.min(scaledLogReco_sortedBySpanet[:,i,1][maskReco])

            else:
                min_pt_eachReco = torch.cat((min_pt_eachReco, torch.Tensor([-1])))
                
        scaledLogReco_sortedBySpanet[...,1] = torch.log(1 + scaledLogReco_sortedBySpanet[...,1])
        mean_pt_cut = torch.mean(scaledLogReco_sortedBySpanet[...,1][maskExist])
        std_pt_cut = torch.std(scaledLogReco_sortedBySpanet[...,1][maskExist])
        meanReco_ptcut = torch.clone(self.meanRecoParticles)
        stdReco_ptcut = torch.clone(self.stdRecoParticles)
        meanReco_ptcut[0] = mean_pt_cut
        stdReco_ptcut[0] = std_pt_cut

        scaledLogReco_sortedBySpanet[...,1] = (scaledLogReco_sortedBySpanet[...,1] - meanReco_ptcut[0])/stdReco_ptcut[0]

        # missing jets -> phi & btag = -1
        scaledLogReco_sortedBySpanet[:,:6,-1] = torch.where(~maskExist[:,:6], -1, scaledLogReco_sortedBySpanet[:,:6,-1])
        scaledLogReco_sortedBySpanet[:,:6,1] = torch.where(~maskExist[:,:6], -1, scaledLogReco_sortedBySpanet[:,:6,1])
        scaledLogReco_sortedBySpanet[:,:6,3] = torch.where(~maskExist[:,:6], -1, scaledLogReco_sortedBySpanet[:,:6,3])

        # padding jets as -100
        scaledLogReco_sortedBySpanet[:,6:,-1] = torch.where(~maskExist[:,6:], -100, scaledLogReco_sortedBySpanet[:,6:,-1])
        scaledLogReco_sortedBySpanet[:,6:,1] = torch.where(~maskExist[:,6:], -100, scaledLogReco_sortedBySpanet[:,6:,1])
        scaledLogReco_sortedBySpanet[:,6:,3] = torch.where(~maskExist[:,6:], -100, scaledLogReco_sortedBySpanet[:,6:,3])

        # check pt != -100
        mask_scaledLogReco_sortedBySpanet = (scaledLogReco_sortedBySpanet[:,:,1] != -100).bool()
        torch.save((scaledLogReco_sortedBySpanet, mask_scaledLogReco_sortedBySpanet, meanReco_ptcut, stdReco_ptcut, min_pt_eachReco),
                    self.processed_file_names('scaledLogReco_sortedBySpanet_ptcut'))

    def sortJets_bySpanet_boxcox(self):
        spanet_tensor = self.scaledLogRecoParticles[:,:16,4:7] # only jets
        spanet_assignment = higgsAssignment_SPANET(spanet_values=spanet_tensor)
        
        # order H1, H2, thad1, thad2, thad3, tlep1, lepton, MET
        scaledLogReco_sortedBySpanet = sortObjects_bySpanet(spanet_assignment=spanet_assignment, scaledLogReco=self.scaledLogRecoParticles_fullCartesian,
                                               maskJets=self.mask_jets, order=[0, 1, 2])

        # at this point the missing Spanet jets and the padding have -100 values
        # 'scaledLogReco_sortedBySpanet' doesn't contain the 'exist' flag for each jet
        exist = torch.where((scaledLogReco_sortedBySpanet[:,:,0] != -100), 1, 0).unsqueeze(dim=2)
        # attach exist flag to each jet
        scaledLogReco_sortedBySpanet = torch.cat((exist, scaledLogReco_sortedBySpanet), dim=2)

        # check exist flag is ok
        if torch.count_nonzero((scaledLogReco_sortedBySpanet[...,0]*scaledLogReco_sortedBySpanet[...,1]) == -100) > 0:
            raise Exception("Check mask... this product must be 0")

        # last part: modify the missing SPANET jets to be [-1...]
        # keep padding jets as [-100...]
        # take the first 6 spanet jets
        scaledLogReco_sortedBySpanet[:,:6,:] = torch.where((scaledLogReco_sortedBySpanet[:,:6,:] == -100), -1, scaledLogReco_sortedBySpanet[:,:6,:])

        # check if the change works (8 because lepton and MET are not -100)
        if torch.count_nonzero(scaledLogReco_sortedBySpanet[:,:8,:] == -100) > 0:
            raise Exception("Missing jets are still -100")

        maskExist = scaledLogReco_sortedBySpanet[...,0] == 1

        # add a column with (torch.logit(btag) - mean)/std
        # save also mean and std
        # I don't care about nans values -> these will be masked during training
        btag_logit = torch.logit(scaledLogReco_sortedBySpanet[...,4], eps=0.001)
        mean_btag = torch.mean(btag_logit[maskExist], dim=0)
        std_btag = torch.std(btag_logit[maskExist], dim=0)
        #btag_logit = (btag_logit - mean_btag)/std_btag
        btag_logit = btag_logit.unsqueeze(dim=2)
        scaledLogReco_sortedBySpanet = torch.cat((scaledLogReco_sortedBySpanet, btag_logit), dim=2)

        # check pt != -100
        mask_scaledLogReco_sortedBySpanet = (scaledLogReco_sortedBySpanet[:,:,1] != -100).bool()

        # unscale pt
        log_pt_unscaled = scaledLogReco_sortedBySpanet[...,1]*self.stdRecoParticles[0] + self.meanRecoParticles[0]

        scaledLogReco_sortedBySpanet[...,1] = torch.where(maskExist, log_pt_unscaled, scaledLogReco_sortedBySpanet[...,1])

        pt_unscaled = torch.exp(scaledLogReco_sortedBySpanet[...,1]) - 1
        scaledLogReco_sortedBySpanet[...,1] = torch.where(maskExist, pt_unscaled, scaledLogReco_sortedBySpanet[...,1])

        # version with pt box cox
        #pt_boxcox, lambda_boxcox = stats.boxcox(scaledLogReco_sortedBySpanet[maskExist][...,1].flatten(), lmbda=0.1757)
        pt_boxcox = stats.boxcox(scaledLogReco_sortedBySpanet[maskExist][...,1].flatten(), lmbda=0.1757)
        lambda_boxcox = 0.1757
        scaledLogReco_sortedBySpanet[maskExist][...,1] = torch.reshape(torch.Tensor(pt_boxcox), (scaledLogReco_sortedBySpanet[maskExist][...,1].shape))

        meanRecoParticles_boxcox = torch.clone(self.meanRecoParticles)
        stdRecoParticles_boxcox = torch.clone(self.stdRecoParticles)

        mean_pt_boxcox = torch.mean(scaledLogReco_sortedBySpanet[maskExist][...,1])
        std_pt_boxcox = torch.std(scaledLogReco_sortedBySpanet[maskExist][...,1])
        meanRecoParticles_boxcox[0] = mean_pt_boxcox
        stdRecoParticles_boxcox[0] = std_pt_boxcox

        pt_scaled_boxcox = (scaledLogReco_sortedBySpanet[...,1] - mean_pt_boxcox)/std_pt_boxcox
        scaledLogReco_sortedBySpanet[...,1] = torch.where(maskExist, pt_scaled_boxcox, scaledLogReco_sortedBySpanet[...,1])

        torch.save((scaledLogReco_sortedBySpanet, mask_scaledLogReco_sortedBySpanet, meanRecoParticles_boxcox, stdRecoParticles_boxcox),
                    self.processed_file_names('scaledLogReco_sortedBySpanet_boxcox'))

        torch.save((torch.Tensor([lambda_boxcox])), self.processed_file_names('lambda_boxcox'))


    def sortJets_byMatched(self):
        prov_tensor = self.scaledLogRecoParticles[:,:16,-1] # only jets
        
        # order H1, H2, thad1, thad2, thad3, tlep1, lepton, MET
        scaledLogReco_sortedByProv = sortObjects_byProv(scaledLogReco=self.scaledLogRecoParticles_fullCartesian,
                                               maskJets=self.mask_jets, order=[0, 1, 2])

        # at this point the missing prov jets and the padding have -100 values
        # 'scaledLogReco_sortedByProv' doesn't contain the 'exist' flag for each jet
        exist = torch.where((scaledLogReco_sortedByProv[:,:,0] != -100), 1, 0).unsqueeze(dim=2)
        # attach exist flag to each jet
        scaledLogReco_sortedByProv = torch.cat((exist, scaledLogReco_sortedByProv), dim=2)

        # check exist flag is ok
        if torch.count_nonzero((scaledLogReco_sortedByProv[...,0]*scaledLogReco_sortedByProv[...,1]) == -100) > 0:
            raise Exception("Check mask... this product must be 0")

        # last part: modify the missing PROV jets to be [-1...]
        # keep padding jets as [-100...]
        # take the first 6 jets
        scaledLogReco_sortedByProv[:,:6,:] = torch.where((scaledLogReco_sortedByProv[:,:6,:] == -100), -1, scaledLogReco_sortedByProv[:,:6,:])

        # check if the change works (8 because lepton and MET are not -100)
        if torch.count_nonzero(scaledLogReco_sortedByProv[:,:8,:] == -100) > 0:
            raise Exception("Missing jets are still -100")

        # check pt != -100
        mask_scaledLogReco_sortedByProv = (scaledLogReco_sortedByProv[:,:,1] != -100).bool()
        torch.save((scaledLogReco_sortedByProv, mask_scaledLogReco_sortedByProv, self.meanRecoParticles, self.stdRecoParticles),
                    self.processed_file_names('scaledLogReco_sortedByProv'))

    def __getitem__(self, index):
        
        if self.debug == True:
            return (self.mask_lepton[index], self.data_lepton[index], self.mask_jets[index],
                    self.data_jets[index], self.mask_met[index], self.data_met[index],
                    self.mask_boost[index], self.data_boost[index],
                    self.scaledLogRecoParticlesCartesian[index], self.LogRecoParticlesCartesian[index])
        
        return [getattr(self, field)[index] if 'mean' not in field and 'std' not in field \
                    else getattr(self, field) for field in self.reco_list]

    def __len__(self):
        size = len(self.mask_lepton)
        return size

