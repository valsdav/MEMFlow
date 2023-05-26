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
import vector

from earlystop import EarlyStopper

torch.cuda.empty_cache()

M_HIGGS = 125.25
M_TOP = 173.29

def check_mass(particle, mean, std):

    particle = particle*std + mean
    particle = torch.sign(particle)*(torch.exp(torch.abs(particle)) - 1)

    particle_try = vector.array(
        {
            "E": particle[:,0].detach().numpy(),
            "px": particle[:,1].detach().numpy(),
            "py": particle[:,2].detach().numpy(),
            "pz": particle[:,3].detach().numpy(),
        }
    )
    
    print(particle_try.mass)

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

def total_mom(higgs, thad, tlep, ISR, mean, std):

    unscaled_higgs = higgs*std[1:] + mean[1:]
    unscaled_thad = thad*std[1:] + mean[1:]
    unscaled_tlep = tlep*std[1:] + mean[1:]
    unscaled_ISR = ISR*std[1:] + mean[1:]

    regressed_higgs = torch.sign(unscaled_higgs)*(torch.exp(torch.abs(unscaled_higgs)) - 1)
    regressed_thad = torch.sign(unscaled_thad)*(torch.exp(torch.abs(unscaled_thad)) - 1)
    regressed_tlep = torch.sign(unscaled_tlep)*(torch.exp(torch.abs(unscaled_tlep)) - 1)
    regressed_ISR = torch.sign(unscaled_ISR)*(torch.exp(torch.abs(unscaled_ISR)) - 1)

    sum_px = regressed_higgs[:,0] + regressed_thad[:,0] + regressed_tlep[:,0] + regressed_ISR[:,0]
    sum_py = regressed_higgs[:,1] + regressed_thad[:,1] + regressed_tlep[:,1] + regressed_ISR[:,1]
    sum_pz = regressed_higgs[:,2] + regressed_thad[:,2] + regressed_tlep[:,2] + regressed_ISR[:,2]

    logsum_px = torch.log(1 + torch.abs(sum_px))
    logsum_py = torch.log(1 + torch.abs(sum_py))
    logsum_pz = torch.log(1 + torch.abs(sum_pz))

    return logsum_px, logsum_py, logsum_pz

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
    early_stopper = EarlyStopper(patience=10, min_delta=1e-5)

    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    regressedPartons_training = torch.zeros((config.training_params.batch_size_training, 4, 4))
    regressedPartons_valid = torch.zeros((config.training_params.batch_size_validation, 4, 4))

    log_mean = torch.tensor(conf.scaling_params.log_mean)
    log_std = torch.tensor(conf.scaling_params.log_std)

    zero_ref = torch.zeros((config.training_params.batch_size_training))

    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.

        sum_px = 0.
        sum_py = 0.
        sum_pz = 0.

        sum_total_p = 0.
    
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

            higgs = out[0]
            thad = out[1]
            tlep = out[2]
            ISR = out[3]

            logsum_px, logsum_py, logsum_pz = total_mom(higgs, thad, tlep, ISR, log_mean, log_std)

            logE_higgs, logE_thad, logE_tlep, logE_ISR = constrain_energy(higgs, thad, tlep, ISR, log_mean, log_std)

            higgs = torch.concat((logE_higgs, higgs), dim=1)
            thad = torch.concat((logE_thad, thad), dim=1)
            tlep = torch.concat((logE_tlep, tlep), dim=1)
            ISR = torch.concat((logE_ISR, ISR), dim=1)

            # check mass for debubgging
            # check_mass(ISR, log_mean, log_std)
            
            losspx = loss_fn(zero_ref, logsum_px)
            losspy = loss_fn(zero_ref, logsum_py)
            losspz = loss_fn(zero_ref, logsum_pz)
        
            lossH = loss_fn(target[:,0], higgs)
            lossThad =  loss_fn(target[:,1], thad)
            lossTlep =  loss_fn(target[:,2], tlep)
            lossISR =  loss_fn(target[:,3], ISR)

            writer.add_scalar('loss_H', lossH.item(), ii)
            writer.add_scalar('loss_Thad', lossThad.item(), ii)
            writer.add_scalar('loss_Tlep', lossTlep.item(), ii)
            writer.add_scalar('loss_ISR', lossISR.item(), ii)
            writer.add_scalar('loss_px', losspx.item(), ii)
            writer.add_scalar('loss_py', losspy.item(), ii)
            writer.add_scalar('loss_pz', losspz.item(), ii)

            loss = lossH + lossThad + lossTlep + lossISR + \
                    0.1*torch.abs(losspx) + 0.1*torch.abs(losspy) + 0.1*torch.abs(losspz)
            
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

        if early_stopper.early_stop(valid_loss, model.state_dict(), optimizer.state_dict(), modelName):
            print(f"Model converges at epoch {e} !!!")         
            break

        scheduler.step() # reduce lr if the model is not improving anymore

    writer.close()

    print('preTraining finished!!')
        

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
    
    
    
    
    
    
    
    
    
    
    
    

