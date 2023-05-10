from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plot_loss
from torch import optim
from tqdm.notebook import trange

from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer

from tensorboardX import SummaryWriter

from torch.nn.functional import normalize
from sklearn.preprocessing import StandardScaler

from memflow.unfolding_flow.utils import *
torch.cuda.empty_cache()

import sys
import argparse
import os

def TrainingAndValidLoop(config, model, trainingLoader, validLoader):
        
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    
    name_dir = f'{os.getcwd()}/results/runs/{conf.name}_{conf.version}'
    writer = SummaryWriter(name_dir)
    
    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    
    ii = 0
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
            scaledLogRecoParticles, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data
            
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            out = model(scaledLogRecoParticles, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
            target = data[0]
        
            lossH = loss_fn(target[:,0], out[0])
            lossThad =  loss_fn(target[:,1], out[1])
            lossTlep =  loss_fn(target[:,2], out[2])
            writer.add_scalar('loss_H', lossH.item(), ii)
            writer.add_scalar('loss_Thad', lossThad.item(), ii)
            writer.add_scalar('loss_Tlep', lossTlep.item(), ii)

            loss = lossH + lossThad + lossTlep
            
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            writer.add_scalar("Loss_step_train", loss.item(), ii)
        
        writer.add_scalar('Loss_epoch_train', sum_loss/N_train, e)
        valid_loss = 0
        valid_lossH = 0
        valid_lossTlep = 0
        valid_lossThad = 0
 
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        for i, data in enumerate(validLoader):
            
            with torch.no_grad():

                (logScaled_data_higgs_t_tbar_ISR_cartesian, 
                scaledLogRecoParticles, mask_lepton_reco, 
                mask_jets, mask_met, 
                mask_boost_reco, data_boost_reco) = data
                
                mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)
        
                out = model(scaledLogRecoParticles, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
                target = data[0]

                lossH = loss_fn(target[:,0], out[0])
                lossThad =  loss_fn(target[:,1], out[1])
                lossTlep =  loss_fn(target[:,2], out[2])

                loss = lossH + lossThad + lossTlep
                valid_loss += loss.item()
                valid_lossH += lossH.item()
                valid_lossTlep += lossTlep.item()
                valid_lossThad += lossThad.item()

                

                if i == 0:
                    for particle in range(3):
                        for feature in range(4):

                            fig, ax = plt.subplots()
                            h = ax.hist2d(out[particle_correct_order][:,feature].cpu().detach().numpy().flatten(),
                                          target[:,particle,feature].detach().cpu().numpy(),
                                          bins=50, range=((0, 1),(0, 1)))
                            fig.colorbar(h[3], ax=ax)
                            writer.add_figure(f"Validation_particleNo_{particle}_Feature_{feature}", fig, e)

                            fig, ax = plt.subplots()
                            h = ax.hist((out[particle_correct_order][:,feature].cpu().detach().numpy().flatten() - \
                                         target[:,particle,feature].detach().cpu().numpy()), bins=100)
                            writer.add_figure(f"Validation_particleNo_{particle}__Feature_{feature}_Diff", fig, e)
         

        writer.add_scalar('Loss_epoch_val', valid_loss/N_valid, e)
        writer.add_scalar('Loss_epoch_val_H', valid_lossH/N_valid, e)
        writer.add_scalar('Loss_epoch_val_Tlep', valid_lossTlep/N_valid, e)
        writer.add_scalar('Loss_epoch_val_Thad', valid_lossThad/N_valid, e)

        writer.close()
        
    resultsDir = f"{os.getcwd()}/results/logs/model_{config.name}_{config.version}"
        
    torch.save({
        'modelA_state_dict': model.state_dict(),
        'optimizerA_state_dict': optimizer.state_dict()
        }, resultsDir)
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU

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
                            reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                        'mask_jets','mask_met',
                                        'mask_boost', 'data_boost'],
                            parton_list=['logScaled_data_higgs_t_tbar_ISR_cartesian'])
    
    # split data for training sample and validation sample
    train_subset, val_subset = torch.utils.data.random_split(
            data_cuda, [conf.training_params.traning_sample, conf.training_params.validation_sample],
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
                                    dtype=torch.float64).cuda()
    
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
            TrainingAndValidLoop(conf, model, train_loader, val_loader)
        else:
            TrainingAndValidLoop(conf, model, train_loader, val_loader)
    else:
        TrainingAndValidLoop(conf, model, train_loader, val_loader)
        
    
    
    
    
    
    
    
    
    
    
    
    
    

