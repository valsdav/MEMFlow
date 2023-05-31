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

M_HIGGS = 125.25
M_TOP = 173.29

def constrain_energy(higgs, thad, tlep, ISR, mean, std):

    unscaled_higgs = higgs*std[1:] + mean[1:]
    unscaled_thad = thad*std[1:] + mean[1:]
    unscaled_tlep = tlep*std[1:] + mean[1:]
    unscaled_ISR = ISR*std[1:] + mean[1:]

    regressed_higgs = torch.sign(unscaled_higgs)*(torch.exp(torch.abs(unscaled_higgs)) - 1)
    regressed_thad = torch.sign(unscaled_thad)*(torch.exp(torch.abs(unscaled_thad)) - 1)
    regressed_tlep = torch.sign(unscaled_tlep)*(torch.exp(torch.abs(unscaled_tlep)) - 1)
    regressed_ISR = torch.sign(unscaled_ISR)*(torch.exp(torch.abs(unscaled_ISR)) - 1)

    E_higgs = torch.sqrt(M_HIGGS**2 + regressed_higgs[:,0]**2 + \
                        regressed_higgs[:,1]**2 + regressed_higgs[:,2]**2).unsqueeze(dim=1)
            
    E_thad = torch.sqrt(M_TOP**2 + regressed_thad[:,0]**2 + \
                        regressed_thad[:,1]**2 + regressed_thad[:,2]**2).unsqueeze(dim=1)

    E_tlep = torch.sqrt(M_TOP**2 + regressed_tlep[:,0]**2 + \
                        regressed_tlep[:,1]**2 + regressed_tlep[:,2]**2).unsqueeze(dim=1)

    E_ISR = torch.sqrt(regressed_ISR[:,0]**2 + regressed_ISR[:,1]**2 + \
                        regressed_ISR[:,2]**2).unsqueeze(dim=1)

    logE_higgs = torch.log(1 + E_higgs)
    logE_thad = torch.log(1 + E_thad)
    logE_tlep = torch.log(1 + E_tlep)
    logE_ISR = torch.log(1 + E_ISR)

    logE_higgs = (logE_higgs - mean[0])/std[0]
    logE_thad = (logE_thad - mean[0])/std[0]
    logE_tlep = (logE_tlep - mean[0])/std[0]
    logE_ISR = (logE_ISR - mean[0])/std[0]

    return logE_higgs, logE_thad, logE_tlep, logE_ISR



def UnscaleTensor(config, model, dataLoader, outputDir):
            
    outputDir = os.path.abspath(outputDir)
            
    total_sample = conf.training_params.training_sample + conf.training_params.validation_sample

    N = len(dataLoader)

    unscaledRegressedPartonsTensor = torch.zeros((total_sample, 4, 4))
    batch_size = 512
    #unscaledRegressedPartons = torch.zeros((batch_size, 4, 4)) # 4096 = batch_size

    log_mean = torch.tensor(conf.scaling_params.log_mean, device=device)
    log_std = torch.tensor(conf.scaling_params.log_std, device=device)
                
    for i, data in enumerate(dataLoader):

        with torch.no_grad():
            if (i % 100 == 0):
                print(i)
                
            (scaledLogRecoParticlesCartesian, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data
                
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            # remove prov
            scaledLogRecoParticlesCartesian = scaledLogRecoParticlesCartesian[:,:,:5]

            out = model(scaledLogRecoParticlesCartesian, data_boost_reco, mask_recoParticles, mask_boost_reco)

            higgs = out[0]
            thad = out[1]
            tlep = out[2]
            ISR = out[3]

            logE_higgs, logE_thad, logE_tlep, logE_ISR = constrain_energy(higgs, thad, tlep, ISR, log_mean, log_std)

            higgs = torch.concat((logE_higgs, higgs), dim=1)
            thad = torch.concat((logE_thad, thad), dim=1)
            tlep = torch.concat((logE_tlep, tlep), dim=1)
            ISR = torch.concat((logE_ISR, ISR), dim=1)

            out_particles = [higgs, thad, tlep, ISR]

            for particle in range(len(out_particles)):
                if out[particle].shape[0] == batch_size:
                    unscaledRegressedPartonsTensor[i*batch_size:(i+1)*batch_size,:,particle] = out_particles[particle]*log_std + log_mean
                else:
                    unscaledRegressedPartonsTensor[i*batch_size:,:,particle] = out_particles[particle]*log_std + log_mean


    
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
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    if (device == torch.device('cuda')):
        env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_var:
            actual_devices = env_var.split(",")
            actual_devices = [int(d) for d in actual_devices]
        else:
            actual_devices = list(range(torch.cuda.device_count()))
        print("Actual devices: ", actual_devices)

    if (device == torch.device('cuda')):

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('')
    
    # READ data
    data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                            reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                        'mask_jets','mask_met',
                                        'mask_boost', 'data_boost'],
                            parton_list=[])
    
    data_loader = DataLoader(dataset=data, shuffle=False, batch_size=512)

    if (device == torch.device('cuda')):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'\ntotal    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        nvmlShutdown()

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('')

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

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('')

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
    
    
    
    
    
    
    
    
    
    
    
    

