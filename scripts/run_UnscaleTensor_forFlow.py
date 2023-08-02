from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from memflow.unfolding_flow.utils import *
from memflow.unfolding_flow.unfolding_flow import UnfoldingFlow

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

def UnscaleTensor(config, model, dataLoader, outputDir):
            
    outputDir = os.path.abspath(outputDir)
            
    total_sample = config.training_params.training_sample + config.training_params.validation_sample

    N = len(dataLoader)

    unscaledRegressedPartonsTensor = torch.empty((total_sample,
                                            config.conditioning_transformer.no_decoders,
                                            config.conditioning_transformer.out_features))
    batch_size = 2048

    log_mean = torch.tensor(config.scaling_params.log_mean, device=device)
    log_std = torch.tensor(config.scaling_params.log_std, device=device)
                
    for i, data in enumerate(dataLoader):

        with torch.no_grad():
            if (i % 100 == 0):
                print(i)
                
            (logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data
                
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            # remove prov
            if (config.noProv):
                logScaled_reco = logScaled_reco[:,:,:-1]

            out = model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)

            for particle in range(len(out)):
                if out[particle].shape[0] == batch_size:
                    unscaledRegressedPartonsTensor[i*batch_size:(i+1)*batch_size,particle,:] = out[particle]
                else:
                    unscaledRegressedPartonsTensor[i*batch_size:,particle,:] = out[particle]

    
    torch.save((unscaledRegressedPartonsTensor), f'{outputDir}/unscaledRegressedPartonsTensor.pt')



        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow-dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    if(args.flow_dir[-1] == '/'):
        args.flow_dir = args.flow_dir[:len(args.flow_dir)-1] # remove '/' from input 

    conf_name = [filename for filename in os.listdir(args.flow_dir) if filename.startswith("config")]

    path_to_conf = f"{args.flow_dir}/{conf_name[0]}"
    model_path = f"{args.flow_dir}/model_flow.pt"
    
    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)

    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.flow_dir
    
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
    if (conf.cartesian):
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=[])
    else:
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=[])
    
    data_loader = DataLoader(dataset=data, shuffle=False, batch_size=2048)

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
    flow_model = UnfoldingFlow(model_path=model_path,
                    read_CondTransf=False,
                    log_mean = conf.scaling_params.log_mean,
                    log_std = conf.scaling_params.log_std,
                    no_jets=conf.input_shape.number_jets,
                    no_lept=conf.input_shape.number_lept,
                    input_features=conf.input_shape.input_features,
                    cond_hiddenFeatures=conf.conditioning_transformer.hidden_features,
                    cond_dimFeedForward=conf.conditioning_transformer.dim_feedforward_transformer,
                    cond_outFeatures=conf.conditioning_transformer.out_features,
                    cond_nheadEncoder=conf.conditioning_transformer.nhead_encoder,
                    cond_NoLayersEncoder=conf.conditioning_transformer.no_layers_encoder,
                    cond_nheadDecoder=conf.conditioning_transformer.nhead_decoder,
                    cond_NoLayersDecoder=conf.conditioning_transformer.no_layers_decoder,
                    cond_NoDecoders=conf.conditioning_transformer.no_decoders,
                    cond_aggregate=conf.conditioning_transformer.aggregate,
                    flow_nfeatures=conf.unfolding_flow.nfeatures,
                    flow_ncond=conf.unfolding_flow.ncond, 
                    flow_ntransforms=conf.unfolding_flow.ntransforms,
                    flow_hiddenMLP_NoLayers=conf.unfolding_flow.hiddenMLP_NoLayers, 
                    flow_hiddenMLP_LayerDim=conf.unfolding_flow.hiddenMLP_LayerDim,
                    flow_bins=conf.unfolding_flow.bins,
                    flow_autoregressive=conf.unfolding_flow.autoregressive,
                    flow_base=conf.unfolding_flow.base,
                    flow_base_first_arg=conf.unfolding_flow.base_first_arg,
                    flow_base_second_arg=conf.unfolding_flow.base_second_arg,
                    flow_bound=conf.unfolding_flow.bound,
                    device=device,
                    dtype=torch.float64)

    state_dict = torch.load(model_path, map_location=device)
    flow_model.load_state_dict(state_dict['model_state_dict'])
    flow_model.eval()

    model = flow_model.cond_transformer
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
    
    
    
    
    
    
    
    
    
    
    
    

