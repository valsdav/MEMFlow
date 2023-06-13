from memflow.unfolding_flow.unfolding_flow import UnfoldingFlow
from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
from earlystop import EarlyStopper
import yaml
from yaml.loader import SafeLoader
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import memflow.unfolding_flow.utils as utils
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.multiprocessing as mp

from torch.cuda.amp import GradScaler
from torch import autocast

import glob
import sys
import argparse
import os

def TrainingAndValidLoop(config, model, trainingLoader, validLoader, outputDir):
    optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    
    name_dir = f'{outputDir}/results'
    modelName = f"{outputDir}/model_flow_{config.version}.pt"
    writer = SummaryWriter(name_dir)

    N_train = len(trainingLoader)
    N_valid = len(validLoader)

    ii = 0
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.001)
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    
    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
    
        # training loop    
        print("Before training loop")
        model.train()
        for i, data in enumerate(trainingLoader):
            
            ii += 1

            if (i % 100 == 0):
                print(i)

            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                logp_g, detjac, cond_X, PS_regressed = model(data, device, config.noProv)

                inf_mask = torch.isinf(logp_g)
                nonzeros = torch.count_nonzero(inf_mask)
                writer.add_scalar(f"Number_INF_flow_{e}", nonzeros.item(), i)

                logp_g = torch.nan_to_num(logp_g, posinf=20, neginf=-20)
                detjac = torch.nan_to_num(detjac, posinf=20, neginf=-20)

                detjac_mean = -detjac.nanmean()
                
                loss = -logp_g.mean()

            scaler.scale(loss).backward()
            #loss.backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_value_(model.parameters(), 20)

            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item()

            writer.add_scalar(f"detJacOnly_epoch_step", detjac_mean.item(), i)
            writer.add_scalar(f"Loss_step_train_epoch_step", loss.item(), i)

        writer.add_scalar('Loss_epoch_train', sum_loss/N_train, e)
        valid_loss = 0

        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        model.eval()
        for i, data in enumerate(validLoader):

            with torch.no_grad():

                logp_g, detjac, cond_X, PS_regressed  = model(data, device, config.noProv)
                logp_g = torch.nan_to_num(logp_g, posinf=20, neginf=-20)
                loss =  -logp_g.mean()
                valid_loss += loss.item()

                if i == 0:

                    (phasespace_intermediateParticles,
                    phasespace_rambo_detjacobian,
                    logScaled_reco, mask_lepton_reco, 
                    mask_jets, mask_met, 
                    mask_boost_reco, data_boost_reco) = data

                    ps_new = model.flow(PS_regressed).sample((config.training_params.sampling_points,))

                    data_ps_cpu = phasespace_intermediateParticles.detach().cpu()
                    ps_new_cpu = ps_new.detach().cpu()

                    for x in range(data_ps_cpu.size(1)):
                        fig, ax = plt.subplots()
                        h = ax.hist2d(data_ps_cpu[:,x].tile(config.training_params.sampling_points,1,1).flatten().numpy(),
                                    ps_new_cpu[:,:,x].flatten().numpy(),
                                    bins=50, range=((0, 1),(0, 1)))
                        fig.colorbar(h[3], ax=ax)
                        writer.add_figure(f"Validation_ramboentry_Plot_{x}", fig, e)

                        fig, ax = plt.subplots()
                        h = ax.hist(
                            (data_ps_cpu[:,x].tile(config.training_params.sampling_points,1,1) - ps_new_cpu[:,:,x]).flatten().numpy(),
                            bins=100)
                        writer.add_figure(f"Validation_ramboentry_Diff_{x}", fig, e)

        writer.add_scalar('Loss_epoch_val', valid_loss/N_valid, e)

        if early_stopper.early_stop(valid_loss, model.state_dict(), optimizer.state_dict(), modelName):
            print(f"Model converges at epoch {e} !!!")         
            break

        scheduler.step() # reduce lr if the model is not improving anymore

    writer.close()
        
    print('Training finished!!')
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    path_to_dir = args.model_dir
    output_dir = f'{args.model_dir}/UnfoldingFlowResult'
    on_GPU = args.on_GPU # by default run on CPU

    path_to_conf = glob.glob(f"{path_to_dir}/*.yaml")[0]
    path_to_model = glob.glob(f"{path_to_dir}/model*.pt")[0]

    print(path_to_conf)
    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
        print(conf)
    
    print("Training with cfg: \n", OmegaConf.to_yaml(conf))

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
    if (conf.cartesian):
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['phasespace_intermediateParticles',
                                            'phasespace_rambo_detjacobian'])
    else:
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['phasespace_intermediateParticles',
                                            'phasespace_rambo_detjacobian'])
    
    # split data for training sample and validation sample
    train_subset, val_subset = torch.utils.data.random_split(
        data, [conf.training_params.training_sample, conf.training_params.validation_sample],
        generator=torch.Generator().manual_seed(1))
    
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=conf.training_params.batch_size_training)
    val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=conf.training_params.batch_size_validation)
    
    # Initialize model
    model = UnfoldingFlow(model_path=path_to_model,
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
                    flow_hiddenMLP_Dim=conf.unfolding_flow.hidden_mlp_dim, 
                    flow_hiddenMLP_Features=conf.unfolding_flow.hidden_mlp_features,
                    flow_bins=conf.unfolding_flow.bins,
                    flow_autoregressive=conf.unfolding_flow.autoregressive,
                    device=device,
                    dtype=torch.float64)
    
    # Copy model on GPU memory
    if (device == torch.device('cuda')):
        model = model.cuda()
        
    print(f"parameters total:{utils.count_parameters(model)}")
    
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
            TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir)
        else:
            TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir)
    else:
        TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir)
        
    print("Training finished succesfully!")
    
    
    
    
    
    
    
    
    
    
    
    
    
