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

def TrainingAndValidLoop(config, model, trainingLoader, validLoader, outputDir):
        
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    outputDir = os.path.abspath(outputDir)
    name_dir = f'{outputDir}/{conf.name}_{conf.version}'
    
    writer = SummaryWriter(name_dir)

    with open(f"{name_dir}/config_{config.name}_{config.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(config)) 
    
    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    
    ii = 0
    trainingLoss_Epoch = [0]*config.training_params.nepochs
    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
    
        # training loop    
        print("Before training loop")
        for i, data in enumerate(trainingLoader):
            
            ii += 1
            if (i % 100 == 0):
                print(i)

            optimizer.zero_grad()
            
            (logScaled_data_higgs_t_tbar_ISR_cartesian, 
            scaledLogRecoParticlesCartesian, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data
            
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            # remove prov
            scaledLogRecoParticlesCartesian = scaledLogRecoParticlesCartesian[:,:,:5]

            out = model(scaledLogRecoParticlesCartesian, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
            target = data[0]
        
            lossH = loss_fn(target[:,0], out[0])
            lossThad =  loss_fn(target[:,1], out[1])
            lossTlep =  loss_fn(target[:,2], out[2])
            lossISR =  loss_fn(target[:,3], out[3])
            writer.add_scalar('loss_H', lossH.item(), ii)
            writer.add_scalar('loss_Thad', lossThad.item(), ii)
            writer.add_scalar('loss_Tlep', lossTlep.item(), ii)
            writer.add_scalar('loss_ISR', lossISR.item(), ii)

            loss = lossH + lossThad + lossTlep + lossISR
            
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            writer.add_scalar("Loss_step_train", loss.item(), ii)
        
        writer.add_scalar('Loss_epoch_train', sum_loss/N_train, e)
        valid_loss = 0
        valid_lossH = 0
        valid_lossTlep = 0
        valid_lossThad = 0
        valid_lossISR = 0

        trainingLoss_Epoch[e] = sum_loss/N_train
        if (e > config.training_params.nEpochsPatience):
            if (abs(trainingLoss_Epoch[e - config.training_params.nEpochsPatience] - trainingLoss_Epoch[e]) < 1e-5):
                print(f"Convergence of loss at epoch: {e}")
                break
 
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        for i, data in enumerate(validLoader):
            
            with torch.no_grad():

                (logScaled_data_higgs_t_tbar_ISR_cartesian, 
                scaledLogRecoParticlesCartesian, mask_lepton_reco, 
                mask_jets, mask_met, 
                mask_boost_reco, data_boost_reco) = data
                
                mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

                # remove prov
                scaledLogRecoParticlesCartesian = scaledLogRecoParticlesCartesian[:,:,:5]
        
                out = model(scaledLogRecoParticlesCartesian, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
                target = data[0]

                lossH = loss_fn(target[:,0], out[0])
                lossThad =  loss_fn(target[:,1], out[1])
                lossTlep =  loss_fn(target[:,2], out[2])
                lossISR =  loss_fn(target[:,3], out[3])

                loss = lossH + lossThad + lossTlep + lossISR
                valid_loss += loss.item()
                valid_lossH += lossH.item()
                valid_lossTlep += lossTlep.item()
                valid_lossThad += lossThad.item()
                valid_lossISR += lossISR.item()

                
                if i == 0:
                    for particle in range(4): # 4 particles: higgs/thad/tlep/gluonISR
                        for feature in range(4):

                            fig, ax = plt.subplots()
                            h = ax.hist2d(out[particle][:,feature].cpu().detach().numpy().flatten(),
                                          target[:,particle,feature].detach().cpu().numpy(),
                                          bins=100, range=((-3, 3),(-3, 3)))
                            fig.colorbar(h[3], ax=ax)
                            writer.add_figure(f"Validation_particleNo_{particle}_Feature_{feature}", fig, e)

                            fig, ax = plt.subplots()
                            h = ax.hist((out[particle][:,feature].cpu().detach().numpy().flatten() - \
                                         target[:,particle,feature].detach().cpu().numpy()), bins=100)
                            writer.add_figure(f"Validation_particleNo_{particle}__Feature_{feature}_Diff", fig, e)
         

        writer.add_scalar('Loss_epoch_val', valid_loss/N_valid, e)
        writer.add_scalar('Loss_epoch_val_H', valid_lossH/N_valid, e)
        writer.add_scalar('Loss_epoch_val_Tlep', valid_lossTlep/N_valid, e)
        writer.add_scalar('Loss_epoch_val_Thad', valid_lossThad/N_valid, e)
        writer.add_scalar('Loss_epoch_val_ISR', valid_lossISR/N_valid, e)

        scheduler.step() # reduce lr if the model is not improving anymore

    writer.close()

    print('preTraining finished, let\'s save the weights')

    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, modelName)

    print('Model saved!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.output_dir

    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
    
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
    data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                            reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                        'mask_jets','mask_met',
                                        'mask_boost', 'data_boost'],
                            parton_list=['logScaled_data_higgs_t_tbar_ISR_cartesian'])
    
    # split data for training sample and validation sample
    train_subset, val_subset = torch.utils.data.random_split(
            data, [conf.training_params.training_sample, conf.training_params.validation_sample],
            generator=torch.Generator().manual_seed(1))
    
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=conf.training_params.batch_size_training)
    val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=conf.training_params.batch_size_validation)

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
            TrainingAndValidLoop(conf, model, train_loader, val_loader, outputDir)
        else:
            TrainingAndValidLoop(conf, model, train_loader, val_loader, outputDir)
    else:
        TrainingAndValidLoop(conf, model, train_loader, val_loader, outputDir)
        
    
    print("PreTraining finished succesfully!")
    
    
    
    
    
    
    
    
    
    
    
    

