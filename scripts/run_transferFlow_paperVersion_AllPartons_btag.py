from comet_ml import Experiment
#from comet_ml.integration.pytorch import log_model

import torch
from memflow.read_data.dataset_all import DatasetCombined
from memflow.transfer_flow.transfer_flow_paper_AllPartons_btag import TransferFlow_Paper_AllPartons_btag
from memflow.unfolding_flow.utils import *
from utils_transferFlow_paper import sample_fullRecoEvent_classifier_AllPartons_btag
from utils_transferFlow_paper import existQuality_print
from utils_transferFlow_paper import sampling_print_btag
from utils_transferFlow_paper import validation_print
from utils_transferFlow_paper import unscale_pt
from utils_transferFlow_paper import compute_loss_per_pt
from utils_transferFlow_paper import plot_grad_flow
from utils_transferFlow_paper import plot_grad_hist

import mdmm

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import floor


# from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import sys
import argparse
import os
from pynvml import *
import vector

from utils import alter_variables

from earlystop import EarlyStopper
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

from torcheval.metrics.functional import multiclass_f1_score

from random import randint
PI = torch.pi

def attach_position(input_tensor, position):
    position = position.expand(input_tensor.shape[0], -1, -1)
    input_tensor = torch.cat((input_tensor, position), dim=2)
    
    return input_tensor

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    import socket
    print("Setting up ddp for device: ", rank)
    os.environ["MASTER_ADDR"] = socket.gethostname()
    os.environ["MASTER_PORT"] = f"{port}"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train( device, name_dir, config,  outputDir, dtype,
           world_size=None, device_ids=None, easy_version=False, min5Jets=False):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device
 
    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    train_dataset = DatasetCombined(config.input_dataset_train, dev=device,
                                    dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['tensor_AllPartons',
                                           'mask_AllPartons'])

    val_dataset = DatasetCombined(config.input_dataset_validation,dev=device,
                                  dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['tensor_AllPartons',
                                           'mask_AllPartons'])

    no_recoObjs = train_dataset.reco_lab.scaledLogReco_sortedBySpanet.shape[1]

    log_mean_reco = train_dataset.reco_lab.meanRecoParticles
    log_std_reco = train_dataset.reco_lab.stdRecoParticles
    log_mean_parton = train_dataset.partons_lab.mean_log_partonsLeptons
    log_std_parton = train_dataset.partons_lab.std_log_partonsLeptons

    if device == torch.device('cuda'):
        log_mean_reco = log_mean_reco.cuda()
        log_std_reco = log_std_reco.cuda()
        log_mean_parton = log_mean_parton.cuda()
        log_std_parton = log_std_parton.cuda()

    if easy_version:
        print('Load easy version: only events with 6/7/8 jets')
        scaledRecoList = [train_dataset.reco_lab.scaledLogReco_sortedBySpanet, val_dataset.reco_lab.scaledLogReco_sortedBySpanet]

        for i in range(2):
            scaledReco = scaledRecoList[i]
            no_objs = torch.sum(scaledReco[:,:,0], dim=1)
            mask_min6Jets_max8Jets = (no_objs >= 8) & (no_objs < 11)
            indices = mask_min6Jets_max8Jets.nonzero().squeeze(dim=1)
            if i == 0:
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            else:
                 val_dataset = torch.utils.data.Subset(val_dataset, indices)

    elif min5Jets:
        print('Load events with min 5 Jets')
        scaledRecoList = [train_dataset.reco_lab.scaledLogReco_sortedBySpanet, val_dataset.reco_lab.scaledLogReco_sortedBySpanet]

        for i in range(2):
            scaledReco = scaledRecoList[i]
            no_objs = torch.sum(scaledReco[:,:,0], dim=1)
            mask_5Jets = (no_objs >= 5)
            indices = mask_5Jets.nonzero().squeeze(dim=1)
            if i == 0:
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            else:
                 val_dataset = torch.utils.data.Subset(val_dataset, indices)

    pt_bins=[5, 50, 75, 100, 150, 200, 300, 1500]

    # Initialize model
    model = TransferFlow_Paper_AllPartons_btag(no_recoVars=6, # exist + 3-mom + encoded_position + btag
                no_partonVars=5,
                no_recoObjects=no_recoObjs,

                no_transformers=config.transformerConditioning.no_transformers,
                transformer_input_features=config.transformerConditioning.input_features,
                transformer_nhead=config.transformerConditioning.nhead,
                transformer_num_encoder_layers=config.transformerConditioning.no_encoder_layers,
                transformer_num_decoder_layers=config.transformerConditioning.no_decoder_layers,
                transformer_dim_feedforward=config.transformerConditioning.dim_feedforward,
                transformer_activation=nn.GELU(),
                 
                flow_nfeatures=4, #pt/eta/phi/btag(logit)
                flow_ntransforms=config.transferFlow.ntransforms,
                flow_hiddenMLP_NoLayers=config.transferFlow.hiddenMLP_NoLayers,
                flow_hiddenMLP_LayerDim=config.transferFlow.hiddenMLP_LayerDim,
                flow_bins=config.transferFlow.bins,
                flow_autoregressive=config.transferFlow.autoregressive,
                flow_base=config.transferFlow.base,
                flow_base_first_arg=config.transferFlow.base_first_arg,
                flow_base_second_arg=config.transferFlow.base_second_arg,
                flow_bound=config.transferFlow.bound,
                randPerm=config.transferFlow.randPerm,
                no_max_objects=config.transferFlow.no_max_objects,

                flow_nfeatures_btag=1,
                flow_ntransforms_btag=config.transferFlow_btag.ntransforms,
                flow_bins_btag=config.transferFlow_btag.bins,
                flow_hiddenMLP_LayerDim_btag=config.transferFlow_btag.hiddenMLP_LayerDim,
                flow_hiddenMLP_NoLayers_btag=config.transferFlow_btag.hiddenMLP_NoLayers,
                flow_base_first_arg_btag=config.transferFlow_btag.base_first_arg,
                flow_base_second_arg_btag=config.transferFlow_btag.base_second_arg,
                flow_base_btag=config.transferFlow_btag.base,
                flow_bound_btag=config.transferFlow_btag.bound,

                flow_lepton_ntransforms=config.transferFlow_lepton.ntransforms,
                flow_lepton_hiddenMLP_NoLayers=config.transferFlow_lepton.hiddenMLP_NoLayers,
                flow_lepton_hiddenMLP_LayerDim=config.transferFlow_lepton.hiddenMLP_LayerDim,
                flow_lepton_bins=config.transferFlow_lepton.bins,

                DNN_nodes=config.DNN.nodes, DNN_layers=config.DNN.layers,
                pretrained_classifier=config.DNN.path_pretraining,
                load_classifier=False,
                encode_position=True,
                 
                device=device,
                dtype=dtype,
                eps=1e-4)

    # Experiment logging
    if device == 0 or world_size is None:
        # Loading comet_ai logging
        exp = Experiment(
            api_key=config.comet_token,
            project_name="MEMFlow",
            workspace="antoniopetre",
            auto_output_logging = "simple",
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True
            # disabled=True
        )
        exp.add_tags(['paper_Impl', 'btag-logit-NotStandardized', 'leptonMETFirst+newFlows+H/thad/tlep', '3DFlow', 'onlyExistJets', f'min5Jets={min5Jets}', 'HiggsAssignment', f'scheduler={config.training_params.scheduler}'])
        exp.log_parameters(config)
        exp.log_parameters({"model_param_tot": count_parameters(model)})
        exp.log_parameters({"model_param_transformer": count_parameters(model.classifier_exist.transformer_model)})
        exp.log_parameters({"model_param_flow_jets": count_parameters(model.flow_kinematics_jets)})
        exp.log_parameters({"model_param_flow_lepton": count_parameters(model.flow_kinematics_lepton)})
        exp.log_parameters({"model_param_flow_MET": count_parameters(model.flow_kinematics_MET)})
        exp.log_parameters({"model_param_flow_btag": count_parameters(model.flow_btag)})
    else:
        exp = None

    # Setting up DDP
    model = model.to(device)

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            # find_unused_parameters=True,
        )
        #print(ddp_model)
        model = ddp_model.module
    else:
        ddp_model = model


    # Datasets
    trainingLoader = DataLoader(
        train_dataset,
        batch_size= config.training_params.batch_size_training,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
    )
    validLoader = DataLoader(
        val_dataset,
        batch_size=config.training_params.batch_size_training,
        shuffle=False,        
    )

    loss_BCE = nn.BCELoss(reduction='none')  

    optimizer = optim.RAdam(list(model.parameters()) , lr=config.training_params.lr)
    # optimizer = optim.Rprop(list(model.parameters()) , lr=config.training_params.lr)
    scheduler_type = config.training_params.scheduler
    
    if scheduler_type == "cosine_scheduler":
        scheduler = CosineAnnealingLR(optimizer,
                                        T_max=config.training_params.cosine_scheduler.Tmax,
                                        eta_min=config.training_params.cosine_scheduler.eta_min)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                factor=config.training_params.reduce_on_plateau.factor,
                                                                patience=config.training_params.reduce_on_plateau.patience,
                                                                threshold=config.training_params.reduce_on_plateau.threshold,
                                                                min_lr=config.training_params.reduce_on_plateau.get("min_lr", 1e-7),
                                                                verbose=True)
    elif scheduler_type == "cyclic_lr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                        base_lr= config.training_params.cyclic_lr.base_lr,
                                                        max_lr= config.training_params.cyclic_lr.max_lr,
                                                        step_size_up=config.training_params.cyclic_lr.step_size_up,
                                                        step_size_down=None,
                                                        cycle_momentum=False,
                                                        gamma=config.training_params.cyclic_lr.gamma,
                                                        mode=config.training_params.cyclic_lr.mode,
                                                        verbose=False)
        
    
    
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.0001)

    # attach one-hot encoded position for jets
    pos_jets_lepton_MET = [pos for pos in range(8)] # 6 jets + lepton + MET
    pos_other_jets = [8 for pos in range(no_recoObjs - 8)]
    
    pos_jets_lepton_MET = torch.tensor(pos_jets_lepton_MET, device=device, dtype=dtype)
    pos_other_jets = torch.tensor(pos_other_jets, device=device, dtype=dtype)
    pos_logScaledReco = torch.cat((pos_jets_lepton_MET, pos_other_jets), dim=0).unsqueeze(dim=1)

    # attach one-hot encoded position for partons
    pos_partons = torch.tensor([pos for pos in range(13)], device=device, dtype=dtype).unsqueeze(dim=1) # higgs, t1, t2, ISR

    # new order: lepton MET higgs1 higgs2 etc
    new_order_list = [6,7,0,1,2,3,4,5]
    lastElems = [i+8 for i in range(no_recoObjs - 8)]
    new_order_list = new_order_list + lastElems
    new_order = torch.LongTensor(new_order_list)

    ii = 0
    ii_valid = 0

    for e in range(config.training_params.nepochs):
        
        N_train = 0
        N_valid = 0
        if world_size is not None:
            print(
                f"[GPU{device_id}] | Rank {device} | Epoch {e} | Batchsize: {config.training_params.batch_size_training*len(device_ids)} | Steps: {len(trainingLoader)}"
            )
            trainingLoader.sampler.set_epoch(e)
            
        sum_loss = 0.
        loss_total_each_object = torch.zeros(config.transferFlow.no_max_objects, device=device)
        loss_per_pt = torch.zeros(len(pt_bins) - 1, device=device)
        total_loss_per_pt = torch.zeros(len(pt_bins) - 1, device=device)
        sum_loss_pt = sum_loss_exist = 0.
    
        # training loop    
        print("Before training loop")
        ddp_model.train()

        for i, data_batch in enumerate(trainingLoader):
            N_train += 1
            ii+=1

            optimizer.zero_grad()
            
            (logScaled_partons,
             mask_partonsLeptons,
             logScaled_reco_sortedBySpanet, mask_recoParticles,
             mask_boost, data_boost_reco) = data_batch
                            
            # exist + 3-mom + btag
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,:,[0,1,2,3,9]]
            # The provenance is remove in the model

            # put the lepton first:
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,new_order,:]
            mask_recoParticles = mask_recoParticles[:,new_order]

            # attach 1 hot-encoded position
            logScaled_reco_sortedBySpanet = attach_position(logScaled_reco_sortedBySpanet, pos_logScaledReco)

            # remove prov from partons
            logScaled_partons = logScaled_partons[:,:,[0,1,2,4]] # [pt,eta,phi,type] -> skip type=1/2 for partons/leptons
            logScaled_partons = attach_position(logScaled_partons, pos_partons)

            avg_flow_prob, flow_prob_batch, flow_prob_jet, flow_prob_btag, flow_prob_MET, flow_prob_lepton, \
            avg_flow_prob_exist, flow_prob_exist_batch, flow_prob_exist = ddp_model(logScaled_reco_sortedBySpanet,
                                                                                logScaled_partons,
                                                                                data_boost_reco,
                                                                                mask_partonsLeptons,
                                                                                mask_recoParticles,
                                                                                mask_boost)


            loss_main_perObj = loss_BCE(flow_prob_exist, logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0])
            #loss_per_batch = -flow_prob_batch + torch.sum(loss_main_perObj*mask_recoParticles, dim=1)
            loss_per_batch = -flow_prob_batch + torch.sum(loss_main_perObj, dim=1)
            
            loss_main = loss_per_batch.mean() 
            flow_prob_lepton = flow_prob_lepton.unsqueeze(dim=1)
            flow_prob_MET = flow_prob_MET.unsqueeze(dim=1)
            
            flow_pr = torch.cat((flow_prob_lepton, flow_prob_MET, flow_prob_jet), dim=1)

            # modify mask_recoParticles: use only jets with exist==1
            mask_recoParticles_exist = logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects, 0] == 1
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects]

            loss_Sum_each_object = torch.sum(-1*flow_pr*mask_recoParticles_exist, dim=0)
            number_MaskedObjects = torch.sum(mask_recoParticles_exist, dim=0)
            loss_mean_each_object = torch.div(loss_Sum_each_object, number_MaskedObjects)

            # for cases when jets no. 12 doesn't exist in any of the events -> nan because 0/0 -> replace with 0
            loss_mean_each_object = torch.nan_to_num(loss_mean_each_object, nan=0.0)
            loss_total_each_object = torch.add(loss_total_each_object, loss_mean_each_object)

            loss_per_pt = compute_loss_per_pt(loss_per_pt, flow_pr, logScaled_reco_sortedBySpanet, mask_recoParticles_exist, log_mean_reco, log_std_reco, config.transferFlow.no_max_objects, pt_bins=pt_bins)
            total_loss_per_pt = torch.add(total_loss_per_pt, loss_per_pt)

            if torch.isnan(loss_total_each_object).any() or torch.isinf(loss_total_each_object).any():
                print(f'ii= {ii} Training_Total: nans = {torch.count_nonzero(torch.isnan(loss_total_each_object))}       infs = {torch.count_nonzero(torch.isinf(loss_total_each_object))}')

            if torch.isnan(total_loss_per_pt).any() or torch.isinf(total_loss_per_pt).any():
                print(f'ii= {ii} Training_pt: nans = {torch.count_nonzero(torch.isnan(total_loss_per_pt))}       infs = {torch.count_nonzero(torch.isinf(total_loss_per_pt))}')

            loss_main.backward()

            if i == 0:

                plot_grad_flow(exp, named_parameters=ddp_model.classifier_exist.transformer_model.named_parameters(),
                               epoch=e, modelName='transformer')
                plot_grad_flow(exp, named_parameters=ddp_model.flow_kinematics_jets.named_parameters(),
                               epoch=e, modelName='flow_jet')
                plot_grad_flow(exp, named_parameters=ddp_model.flow_kinematics_lepton.named_parameters(),
                               epoch=e, modelName='flow_lepton')
                plot_grad_flow(exp, named_parameters=ddp_model.flow_kinematics_MET.named_parameters(),
                               epoch=e, modelName='flow_MET')

                plot_grad_hist(exp, named_parameters=ddp_model.classifier_exist.transformer_model.named_parameters(),
                           epoch=e, modelName='grad_transformer')
                plot_grad_hist(exp, named_parameters=ddp_model.flow_kinematics_jets.named_parameters(),
                           epoch=e, modelName='flow_jet')
                plot_grad_hist(exp, named_parameters=ddp_model.flow_kinematics_lepton.named_parameters(),
                           epoch=e, modelName='flow_lepton')
                plot_grad_hist(exp, named_parameters=ddp_model.flow_kinematics_MET.named_parameters(),
                           epoch=e, modelName='flow_MET')
                
                
            optimizer.step()
            sum_loss += loss_main.item()
            #sum_loss_exist += torch.sum(loss_main_perObj*mask_recoParticles, dim=1).mean()
            sum_loss_exist += torch.sum(loss_main_perObj, dim=1).mean()

            if scheduler_type == "cyclic_lr": #cycle each step
                scheduler.step()

            with torch.no_grad():
                if exp is not None and device==0 or world_size is None:
                    if i % config.training_params.interval_logging_steps == 0:
                        exp.log_metric('loss_step', loss_main, step=ii)
                        #exp.log_metric('loss_step_exist', torch.sum(loss_main_perObj*mask_recoParticles, dim=1).mean(), step=ii)
                        exp.log_metric('loss_step_exist', torch.sum(loss_main_perObj, dim=1).mean(), step=ii)
                        exp.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=ii)
                        for j in range(config.transferFlow.no_max_objects - 1):
                            exp.log_metric(f'loss_object_{j}', loss_mean_each_object[j], step=ii)
                        for j in range(len(pt_bins) - 1):
                            exp.log_metric(f'loss_per_pt_{pt_bins[j]}_{pt_bins[j+1]}', loss_per_pt[j], step=ii)

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_train_epoch", sum_loss/N_train, epoch=e)
            exp.log_metric("loss_train_epoch_exist", sum_loss_exist/N_train, epoch=e)
            for j in range(config.transferFlow.no_max_objects - 1):
                exp.log_metric(f'loss_epoch_total_object_{j}', loss_total_each_object[j]/N_train, epoch=e)
            for j in range(len(pt_bins) - 1):
                exp.log_metric(f'loss_epoch_total_per_pt_{pt_bins[j]}_{pt_bins[j+1]}', total_loss_per_pt[j]/N_train, epoch=e)
            # if scheduler_type == "cyclic_lr":
            #     exp.log_metric("learning_rate", scheduler.get_last_lr(), epoch=e, step=ii)

        total_valid_loss = 0.
        loss_Valid_total_each_object = torch.zeros(config.transferFlow.no_max_objects, device=device)
        valid_total_loss_per_pt = torch.zeros(len(pt_bins) - 1, device=device)
        sum_valid_loss_pt = sum_valid_loss_eta = sum_valid_loss_phi = sum_valid_loss_exist = 0.
        f1_valid_total = 0.
        f1_valid_total_eachObject = torch.zeros(config.transferFlow.no_max_objects, device=device)
        
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        ddp_model.eval()
        
        for i, data_batch in enumerate(validLoader):
            N_valid += 1
            ii_valid += 1
            # Move data to device
            with torch.no_grad():

                (logScaled_partons,
                 mask_partonsLeptons,
                 logScaled_reco_sortedBySpanet, mask_recoParticles,
                 mask_boost, data_boost_reco) = data_batch
                                
                # exist + 3-mom + logit(btag)
                logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,:,[0,1,2,3,9]]
                # The provenance is remove in the model
    
                # put the lepton first:
                logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,new_order,:]
                mask_recoParticles = mask_recoParticles[:,new_order]
    
                # attach 1 hot-encoded position
                logScaled_reco_sortedBySpanet = attach_position(logScaled_reco_sortedBySpanet, pos_logScaledReco)
    
                # remove prov from partons
                logScaled_partons = logScaled_partons[:,:,[0,1,2,4]] # [pt,eta,phi,type] -> skip type=1/2 for partons/leptons
                logScaled_partons = attach_position(logScaled_partons, pos_partons)

                # The provenance is remove in the model
                avg_flow_prob, flow_prob_batch, flow_prob_jet, flow_prob_btag, flow_prob_MET, flow_prob_lepton, \
                avg_flow_prob_exist, flow_prob_exist_batch, flow_prob_exist = ddp_model(logScaled_reco_sortedBySpanet,
                                                                                    logScaled_partons,
                                                                                    data_boost_reco,
                                                                                    mask_partonsLeptons,
                                                                                    mask_recoParticles,
                                                                                    mask_boost)
    

                loss_main_perObj = loss_BCE(flow_prob_exist, logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0])
                #loss_per_batch = -flow_prob_batch + torch.sum(loss_main_perObj*mask_recoParticles, dim=1)
                loss_per_batch = -flow_prob_batch + torch.sum(loss_main_perObj, dim=1) # negative log likelihood
                
                loss_main = loss_per_batch.mean() 
                flow_prob_lepton = flow_prob_lepton.unsqueeze(dim=1)
                flow_prob_MET = flow_prob_MET.unsqueeze(dim=1)
                
                flow_pr = torch.cat((flow_prob_lepton, flow_prob_MET, flow_prob_jet), dim=1)
                batch_flow_pr = -loss_per_batch # positive log likelihood
                
                    
                total_valid_loss += loss_main.item() # using only the main loss
                #sum_valid_loss_exist += torch.sum(loss_main_perObj*mask_recoParticles, dim=1).mean()
                sum_valid_loss_exist += torch.sum(loss_main_perObj, dim=1).mean()

                mask_recoParticles_exist = logScaled_reco_sortedBySpanet[:,:,0] == 1

                loss_Sum_each_object = torch.sum(-1*flow_pr*mask_recoParticles_exist[:,:config.transferFlow.no_max_objects], dim=0)
                number_MaskedObjects = torch.sum(mask_recoParticles_exist[:,:config.transferFlow.no_max_objects], dim=0)
                loss_mean_each_object = torch.div(loss_Sum_each_object, number_MaskedObjects)

                loss_mean_each_object = torch.nan_to_num(loss_mean_each_object, nan=0.0)
                loss_Valid_total_each_object = torch.add(loss_Valid_total_each_object, loss_mean_each_object)

                loss_per_pt = compute_loss_per_pt(loss_per_pt, flow_pr, logScaled_reco_sortedBySpanet, mask_recoParticles_exist, log_mean_reco, log_std_reco, config.transferFlow.no_max_objects, pt_bins=pt_bins)

                valid_total_loss_per_pt = torch.add(valid_total_loss_per_pt, loss_per_pt)

                if torch.isnan(loss_Valid_total_each_object).any() or torch.isinf(loss_Valid_total_each_object).any():
                    print(f'ii= {ii} VALID_total: nans = {torch.count_nonzero(torch.isnan(loss_Valid_total_each_object))}       infs = {torch.count_nonzero(torch.isinf(loss_Valid_total_each_object))}')

                if torch.isnan(valid_total_loss_per_pt).any() or torch.isinf(valid_total_loss_per_pt).any():
                    print(f'ii= {ii} VALID_pt: nans = {torch.count_nonzero(torch.isnan(valid_total_loss_per_pt))}       infs = {torch.count_nonzero(torch.isinf(valid_total_loss_per_pt))}')

                exp.log_metric('validation_loss_step_exist', avg_flow_prob_exist, step=ii)

                prediction = torch.where(flow_prob_exist < 0.5, 0, 1)
                f1_score = multiclass_f1_score(prediction.flatten(), logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0].flatten(), num_classes=2)

                f1_valid_total += f1_score
                for jet in range(config.transferFlow.no_max_objects):
                    f1_valid_total_eachObject[jet] += multiclass_f1_score(prediction[:,jet], logScaled_reco_sortedBySpanet[:,jet,0], num_classes=2)

                
                if i == 0 and False:
                    random = torch.rand(1)
                    sign_1 = 1 if random[0] > 0.5 else -1
                    # TODO ADD 10 % of eta
                    difference_eta = [sign_1*0.1]

                    # Jets validation
                    # difference_pt = 10% of first jet
                    difference_pt = 1/10*unscale_pt(logScaled_reco_sortedBySpanet[:,:,1], mask_recoParticles[:,:],
                                                    log_mean_reco, log_std_reco)[:,0]
                    
                
                    # wrong pt
                    wrong_logScaled_reco = alter_variables(difference=difference_pt,
                                            object_no=[0],
                                            variable_altered=[1], # pt = 1 for reco objs
                                            target_var=logScaled_reco_sortedBySpanet, 
                                            log_mean=log_mean_reco, 
                                            log_std=log_std_reco,
                                            mask_target=mask_recoParticles,
                                            no_max_objects=config.transferFlow.no_max_objects,
                                            device=device,
                                            reco=1)
                                        
                    wrongPT_avg_flow_prob_jet, wrongPT_flow_prob_jet_batch, wrongPT_flow_prob_jet, \
                    wrongPT_flow_prob_btag, \
                    wrongPT_flow_prob_MET, wrongPT_flow_prob_lepton, \
                    wrongPT_avg_flow_prob_exist, wrongPT_flow_prob_exist_batch, wrongPT_flow_prob_exist= ddp_model(wrong_logScaled_reco,
                                                                                                                logScaled_partons,
                                                                                                                data_boost_reco,
mask_partonsLeptons,
                                                                                                                   mask_recoParticles,
                                                                                                                mask_boost)
                    
                    wrong_pt_loss_main_perObj = loss_BCE(wrongPT_flow_prob_exist, logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0])
                    #wrong_pt_batch_flow_pr = -wrongPT_flow_prob_jet_batch + torch.sum(wrong_pt_loss_main_perObj*mask_recoParticles, dim=1)
                    wrong_pt_batch_flow_pr = -wrongPT_flow_prob_jet_batch + torch.sum(wrong_pt_loss_main_perObj, dim=1)
                    wrong_pt_batch_flow_pr = -1*wrong_pt_batch_flow_pr # positive log likelihood

                    # wrong pt and eta
                    wrong_logScaled_reco = alter_variables(difference=difference_eta, 
                                            object_no=[1],
                                            variable_altered=[2], # eta = 2 for reco objs
                                            target_var=wrong_logScaled_reco,
                                            log_mean=log_mean_reco, 
                                            log_std=log_std_reco,
                                            mask_target=mask_recoParticles,
                                            no_max_objects=config.transferFlow.no_max_objects,
                                            device=device,
                                            reco=1)
                    
                    wrongPTandETA_avg_flow_prob_jet, wrongPTandETA_flow_prob_jet_batch, wrongPTandETA_flow_prob_jet, \
                    wrongPTandETA_flow_prob_btag, \
                    wrongPTandETA_flow_prob_MET, wrongPTandETA_flow_prob_lepton, \
                    wrongPTandETA_avg_flow_prob_exist, wrongPTandETA_flow_prob_exist_batch, wrongPTandETA_flow_prob_exist = ddp_model(wrong_logScaled_reco,
                                                                                                            logScaled_partons,
                                                                                                            data_boost_reco,
                                                                                                                    
                                                                                                                    mask_partonsLeptons,
                                                                                                            mask_recoParticles,
                                                                                                            mask_boost)
                    
                    wrong_ptAndEta_loss_main_perObj = loss_BCE(wrongPTandETA_flow_prob_exist, logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0])
                    #wrong_ptAndEta_batch_flow_pr = -wrongPTandETA_flow_prob_jet_batch + torch.sum(wrong_ptAndEta_loss_main_perObj*mask_recoParticles, dim=1)
                    wrong_ptAndEta_batch_flow_pr = -wrongPTandETA_flow_prob_jet_batch + torch.sum(wrong_ptAndEta_loss_main_perObj, dim=1)
                    wrong_ptAndEta_batch_flow_pr = -1*wrong_ptAndEta_batch_flow_pr

                    # sometimes there are nans if the difference_pt is too large
                    if torch.isnan(wrong_pt_batch_flow_pr).any() or torch.isnan(wrong_ptAndEta_batch_flow_pr).any():
                        print(f'validation_plots_nans: wrong_pt = {torch.count_nonzero(torch.isnan(wrong_pt_batch_flow_pr))}     & wrong_pt_eta = {torch.count_nonzero(torch.isnan(wrong_ptAndEta_batch_flow_pr))}')

                    # print jet validation
                    validation_print(experiment=exp, flow_pr=batch_flow_pr,
                                    wrong_pt_batch_flow_pr=wrong_pt_batch_flow_pr,
                                    wrong_ptAndEta_batch_flow_pr=wrong_ptAndEta_batch_flow_pr, epoch=e,
                                    range_x=(-60,60), no_bins=120, label1='diff: pt_0 10%',
                                    label2=f'diff: pt_0 10% and eta {difference_eta}', particles='jets')

                    # Partons validation
                    # difference_pt = 10% of Higgs
                    maskPartons = torch.ones(logScaled_partons[:,:,0].shape, device=device)
                    difference_pt = 1/10*unscale_pt(logScaled_partons[...,0], maskPartons,
                                                    log_mean_parton, log_std_parton)[:,0]

                    # wrong pt
                    wrong_logScaled_parton = alter_variables(difference=difference_pt,
                                            object_no=[0],
                                            variable_altered=[0], # pt = 0 for partons
                                            target_var=logScaled_partons, 
                                            log_mean=log_mean_parton, 
                                            log_std=log_std_parton,
                                            mask_target=maskPartons,
                                            no_max_objects=4,
                                            device=device,
                                            reco=0)
                                        
                    wrongPT_avg_flow_prob_jet, wrongPT_flow_prob_jet_batch, wrongPT_flow_prob_jet, \
                    wrongPT_flow_prob_btag, \
                    wrongPT_flow_prob_MET, wrongPT_flow_prob_lepton, \
                    wrongPT_avg_flow_prob_exist, wrongPT_flow_prob_exist_batch, wrongPT_flow_prob_exist = ddp_model(logScaled_reco_sortedBySpanet,
                                                                                                                wrong_logScaled_parton,
                                                                                                                data_boost_reco,
                                                                                                                    mask_partonsLeptons,
                                                                                                                mask_recoParticles,
                                                                                                                mask_boost)

                    wrong_pt_loss_main_perObj = loss_BCE(wrongPT_flow_prob_exist, logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0])
                    #wrong_pt_batch_parton_flow_pr = -wrongPT_flow_prob_jet_batch + torch.sum(wrong_pt_loss_main_perObj*mask_recoParticles, dim=1)
                    wrong_pt_batch_parton_flow_pr = -wrongPT_flow_prob_jet_batch + torch.sum(wrong_pt_loss_main_perObj, dim=1)
                    wrong_pt_batch_parton_flow_pr = -1*wrong_pt_batch_parton_flow_pr

                    # wrong pt and eta
                    wrong_logScaled_parton = alter_variables(difference=difference_eta, 
                                            object_no=[1],
                                            variable_altered=[1], # eta = 1 for partons
                                            target_var=wrong_logScaled_parton,
                                            log_mean=log_mean_parton, 
                                            log_std=log_std_parton,
                                            mask_target=maskPartons,
                                            no_max_objects=4,
                                            device=device,
                                            reco=0)
                    
                    wrongPTandETA_avg_flow_prob_jet, wrongPTandETA_flow_prob_jet_batch, wrongPTandETA_flow_prob_jet, \
                    wrongPTandETA_flow_prob_btag, \
                    wrongPTandETA_flow_prob_MET, wrongPTandETA_flow_prob_lepton, \
                    wrongPTandETA_avg_flow_prob_exist, wrongPTandETA_flow_prob_exist_batch, wrongPTandETA_flow_prob_exist = ddp_model(logScaled_reco_sortedBySpanet,
                                                                                                            wrong_logScaled_parton,
                                                                                                            data_boost_reco,
                                                                                                                    mask_partonsLeptons,
                                                                                                            mask_recoParticles,
                                                                                                            mask_boost)
                    
                    wrong_ptAndEta_loss_main_perObj = loss_BCE(wrongPTandETA_flow_prob_exist, logScaled_reco_sortedBySpanet[:,:config.transferFlow.no_max_objects,0])
                    #wrong_ptAndEta_batch_parton_flow_pr = -wrongPTandETA_flow_prob_jet_batch + torch.sum(wrong_ptAndEta_loss_main_perObj*mask_recoParticles, dim=1)
                    wrong_ptAndEta_batch_parton_flow_pr = -wrongPTandETA_flow_prob_jet_batch + torch.sum(wrong_ptAndEta_loss_main_perObj, dim=1)
                    wrong_ptAndEta_batch_parton_flow_pr = -1*wrong_ptAndEta_batch_parton_flow_pr

                    # sometimes there are nans if the difference_pt is too large
                    if torch.isnan(wrong_pt_batch_parton_flow_pr).any() or torch.isnan(wrong_ptAndEta_batch_parton_flow_pr).any():
                        print(f'validation_plots_nans: wrong_pt = {torch.count_nonzero(torch.isnan(wrong_pt_batch_parton_flow_pr))}     & wrong_pt_eta = {torch.count_nonzero(torch.isnan(wrong_ptAndEta_batch_parton_flow_pr))}')

                    # print parton validation
                    validation_print(experiment=exp, flow_pr=batch_flow_pr,
                                    wrong_pt_batch_flow_pr=wrong_pt_batch_parton_flow_pr,
                                    wrong_ptAndEta_batch_flow_pr=wrong_ptAndEta_batch_parton_flow_pr, epoch=e,
                                    range_x=(-60,60), no_bins=120, label1='diff: pt_0 10%',
                                    label2=f'diff: pt_0 10% and eta {difference_eta}', particles='partons')

                if i == 0 and e % 2 == 0:
                        
                    # print sampled partons
                    fullGeneratedEvent, mask_reco = sample_fullRecoEvent_classifier_AllPartons_btag(model, logScaled_partons, mask_partonsLeptons, logScaled_reco_sortedBySpanet.shape[0], device, dtype, No_samples=1)
        
                    allJets = [i for i in range(config.transferFlow.no_max_objects)]
                    sampling_print_btag(exp, fullGeneratedEvent, logScaled_reco_sortedBySpanet, mask_recoParticles, allJets, e, onlyExistElem=True)
                    existQuality_print(exp, fullGeneratedEvent[:,:,0], logScaled_reco_sortedBySpanet, allJets, e)
    
                    for jet in range(config.transferFlow.no_max_objects):
                        sampling_print_btag(exp, fullGeneratedEvent, logScaled_reco_sortedBySpanet, mask_recoParticles, jet, e, onlyExistElem=True)
                        existQuality_print(exp, fullGeneratedEvent[:,:,0], logScaled_reco_sortedBySpanet, jet, e)
                    
                                                    

        if exp is not None and device==0 or world_size is None:
            exp.log_metric("total_valid_loss", total_valid_loss/N_valid, epoch=e)
            exp.log_metric("total_valid_loss_exist", sum_valid_loss_exist/N_valid, epoch=e)
            exp.log_metric("f1_valid_total_classifier_exist", f1_valid_total/N_valid, epoch=e)
            for j in range(config.transferFlow.no_max_objects - 1):
                exp.log_metric(f'loss_Valid_epoch_total_object_{j}', loss_Valid_total_each_object[j]/N_valid, epoch=e)
                exp.log_metric(f'f1_valid_total_classifierExist_eachObject_{j}', f1_valid_total_eachObject[j]/N_valid, epoch=e)
            for j in range(len(pt_bins) - 1):
                exp.log_metric(f'loss_Valid_epoch_total_per_pt_{pt_bins[j]}_{pt_bins[j+1]}', valid_total_loss_per_pt[j]/N_valid, epoch=e)
            
        if device == 0 or world_size is None:
            if early_stopper.early_stop(total_valid_loss/N_valid,
                                    model.state_dict(), optimizer.state_dict(), modelName, exp):
                print(f"Model converges at epoch {e} !!!")         
                break

        # Step the scheduler at the end of the val
        if scheduler_type == "cosine_scheduler":
            after_N_epochs = config.training_params.cosine_scheduler.get("after_N_epochs", 0)
            if e > after_N_epochs:
                scheduler.step()
                
        elif scheduler_type == "reduce_on_plateau":
            # Step the scheduler at the end of the val
            scheduler.step(total_valid_loss/N_valid)

    # exp_log.end()
    destroy_process_group()
    print('Flow training finished!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--easy-version', action="store_true",  help='6 Jets')
    parser.add_argument('--min5Jets', action="store_true",  help='min5Jets')
    parser.add_argument('--distributed', action="store_true")
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.output_dir
    easy_version = args.easy_version
    min5Jets = args.min5Jets

    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
    
    print("Training with cfg: \n", OmegaConf.to_yaml(conf))

    env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_var:
        actual_devices = env_var.split(",")
    else:
        actual_devices = list(range(torch.cuda.device_count()))
    print("Actual devices: ", actual_devices)
    world_size = len(actual_devices)

    outputDir = os.path.abspath(outputDir)
    name_dir = f'{outputDir}/Transfer_Flow_Paper_AllPartons_btag_scaled_min5Jets={min5Jets}_{conf.name}_{conf.version}_{conf.transferFlow.base}_NoTransf{conf.transferFlow.ntransforms}_NoBins{conf.transferFlow.bins}_DNN:{conf.transferFlow.hiddenMLP_NoLayers}_{conf.transferFlow.hiddenMLP_LayerDim}'
    

    os.makedirs(name_dir, exist_ok=True)
    
    with open(f"{name_dir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf)) 

    if conf.training_params.dtype == "float32":
        dtype = torch.float32
    elif conf.training_params.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = None
        
    
    if len(actual_devices) > 1 and args.distributed:
        # make a dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        print(f"Devices dict: {dev_dct}")
        mp.spawn(
            train,
            args=(name_dir, conf,  outputDir, dtype,
                    world_size, dev_dct, easy_version, min5Jets),
            nprocs=world_size,
            # join=True
        )
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train(device, name_dir, conf,  outputDir, dtype, None, None, easy_version, min5Jets)
    
    print(f"Flow training finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
