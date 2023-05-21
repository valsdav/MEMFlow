from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from memflow.unfolding_flow.utils import *

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import sys
import argparse
import os
from pynvml import *

torch.cuda.empty_cache()

def UnscaleTensor(config, model, dataLoader, outputDir):
            
    outputDir = os.path.abspath(outputDir)
            
    total_sample = conf.training_params.training_sample + conf.training_params.validation_sample

    N = len(dataLoader)

    unscaledRegressedPartonsTensor = torch.zeros((total_sample, 4, 4))
    batch_size = 256
    unscaledRegressedPartons = torch.zeros((batch_size, 4, 4)) # 4096 = batch_size
    print(f"unscaledRegressedPartonsTensor: {unscaledRegressedPartonsTensor.shape}")
                
    for i, data in enumerate(dataLoader):

        if (i % 100 == 0):
            print(i)
            
        (logScaled_data_higgs_t_tbar_ISR_cartesian,
        mean_log_data_higgs_t_tbar_ISR_cartesian,
        std_log_data_higgs_t_tbar_ISR_cartesian,
        scaledLogRecoParticlesCartesian, mask_lepton_reco, 
        mask_jets, mask_met, 
        mask_boost_reco, data_boost_reco) = data
            
        mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

        # remove prov
        #scaledLogRecoParticlesCartesian = scaledLogRecoParticlesCartesian[:,:,:5]

        out = model(scaledLogRecoParticlesCartesian, data_boost_reco, mask_recoParticles, mask_boost_reco)

        for particle in range(len(out)):
            unscaledRegressedPartons[0:out[particle].shape[0], particle] = out[particle]*std_log_data_higgs_t_tbar_ISR_cartesian[0:out[particle].shape[0],:] \
                                                + mean_log_data_higgs_t_tbar_ISR_cartesian[0:out[particle].shape[0],:]

        if out[particle].shape[0] == batch_size:
            unscaledRegressedPartonsTensor[i*batch_size:(i+1)*batch_size,:,:] = unscaledRegressedPartons
        else:
            unscaledRegressedPartonsTensor[i*batch_size:,:,:] = unscaledRegressedPartons[0:out[particle].shape[0], :]

    
    torch.save((unscaledRegressedPartonsTensor), f'{outputDir}/unscaledRegressedPartonsTensor.pt')



        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    if(args.model_dir[-1] != '/'):
        raise("--model-dir argument must be directory, finishing in `/`")
    else:
        args.model_dir = args.model_dir[:len(args.model_dir)-1] # remove '/' from input 

    dirName = os.path.basename(args.model_dir) # get 'directory name'
    path_to_conf = f"{args.model_dir}/config_{dirName}.yaml"
    model_path = f"{args.model_dir}/model_{dirName}.pt"
    
    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)

    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.model_dir
    
    if on_GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    if (device == torch.device('cuda')):
        torch.cuda.empty_cache()
        env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_var:
            actual_devices = env_var.split(",")
            actual_devices = [int(d) for d in actual_devices]
        else:
            actual_devices = list(range(torch.cuda.device_count()))
        print("Actual devices: ", actual_devices)
    
    # READ data
    data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                            reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                        'mask_jets','mask_met',
                                        'mask_boost', 'data_boost'],
                            parton_list=['logScaled_data_higgs_t_tbar_ISR_cartesian',
                                        'mean_log_data_higgs_t_tbar_ISR_cartesian',
                                        'std_log_data_higgs_t_tbar_ISR_cartesian'])
    
    data_loader = DataLoader(dataset=data, shuffle=False, batch_size=256)

    # Initialize model
    model = ConditioningTransformerLayer(no_jets = conf.input_shape.number_jets,
                                    no_lept = conf.input_shape.number_lept,
                                    input_features=conf.input_shape.input_features, 
                                    hidden_features=conf.conditioning_transformer.hidden_features,
                                    dim_feedforward_transformer= conf.conditioning_transformer.dim_feedforward_transformer,
                                    out_features=conf.conditioning_transformer.out_features,
                                    nhead_encoder=conf.conditioning_transformer.nhead_encoder,
                                    no_layers_encoder=conf.conditioning_transformer.no_layers_encoder,
                                    nhead_decoder=conf.conditioning_transformer.nhead_decoder,
                                    no_layers_decoder=conf.conditioning_transformer.no_layers_decoder,
                                    no_decoders=conf.conditioning_transformer.no_decoders,
                                    aggregate=conf.conditioning_transformer.aggregate,
                                    dtype=torch.float64)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    
    if (device == torch.device('cuda')):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'\ntotal    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        nvmlShutdown()

    # Copy model on GPU memory
    if (device == torch.device('cuda')):
        model = model.cuda()

    print(f"parameters total:{count_parameters(model)}")
    if (device == torch.device('cuda')):
        
        # TODO: split the data for multi-GPU processing
        if len(actual_devices) > 1:
            #world_size = torch.cuda.device_count()
            # make a dictionary with k: rank, v: actual device
            #dev_dct = {i: actual_devices[i] for i in range(world_size)}
            #print(f"Devices dict: {dev_dct}")
            #mp.spawn(
            #    TrainingAndValidLoop,
            #    args=(conf, model, train_loader, val_loader, world_size),
            #    nprocs=world_size,
            #    join=True,
            #)
            UnscaleTensor(conf, model, data_loader, outputDir)
        else:
            UnscaleTensor(conf, model, data_loader, outputDir)
    else:
        UnscaleTensor(conf, model, data_loader, outputDir)
        
    
    print("Unscale tensor finished!")
    
    
    
    
    
    
    
    
    
    
    
    

