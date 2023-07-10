import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
import zuko
from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

from memflow.unfolding_flow.mmd_loss import MMDLoss
from memflow.unfolding_flow.mmd_loss import RBF

def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    


class UnfoldingFlow(nn.Module):
    def __init__(self, model_path, log_mean, log_std, no_jets, no_lept, input_features, cond_hiddenFeatures=64,
                cond_dimFeedForward=512, cond_outFeatures=32, cond_nheadEncoder=4, cond_NoLayersEncoder=2,
                cond_nheadDecoder=4, cond_NoLayersDecoder=2, cond_NoDecoders=3, cond_aggregate=False,
                flow_nfeatures=12, flow_ncond=34, flow_ntransforms=5, flow_hiddenMLP_NoLayers=16,
                flow_hiddenMLP_LayerDim=128, flow_bins=16, flow_autoregressive=True, 
                flow_base=BoxUniform, flow_base_first_arg=-1, flow_base_second_arg=1, flow_bound=1.,
                device=torch.device('cpu'), dtype=torch.float64):

        super(UnfoldingFlow, self).__init__()

        self.log_mean = torch.tensor(log_mean, device=device)
        self.log_std = torch.tensor(log_std, device=device)
        
        self.cond_transformer = ConditioningTransformerLayer(
                                    no_jets = no_jets,
                                    no_lept = no_lept,
                                    input_features=input_features,
                                    hidden_features=cond_hiddenFeatures,
                                    dim_feedforward_transformer=cond_dimFeedForward,
                                    out_features=cond_outFeatures,
                                    nhead_encoder=cond_nheadEncoder,
                                    no_layers_encoder=cond_NoLayersEncoder,
                                    nhead_decoder=cond_nheadDecoder,
                                    no_layers_decoder=cond_NoLayersDecoder,
                                    no_decoders=cond_NoDecoders,
                                    aggregate=cond_aggregate,
                                    dtype=dtype)

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.cond_transformer.load_state_dict(state_dict['model_state_dict'])   
        
        self.flow = zuko.flows.NSF(features=flow_nfeatures,
                              context=flow_ncond, 
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=False,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-1,1] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.flow.transforms.insert(0, SimpleAffineTransform(0*torch.ones(flow_nfeatures),1*torch.ones(flow_nfeatures),
                                                     -1*torch.ones(flow_nfeatures), 1*torch.ones(flow_nfeatures)))

        kernel = RBF(device=device)
        self.mmdLoss = MMDLoss(kernel=kernel)
        
        
    def forward(self, data, device, noProv, sampling_Forward=False, eps=0.0, order=[0,1,2,3]):

        (PS_target,
        PS_rambo_detjacobian,
        logScaled_reco, mask_lepton_reco, 
        mask_jets, mask_met, 
        mask_boost_reco, data_boost_reco) = data

        mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

        if (noProv):
            logScaled_reco = logScaled_reco[:,:,:-1]
        
        cond_X = self.cond_transformer(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)
        HttISR_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_numpy(cond_X, self.log_mean,
                                                                                    self.log_std, device, eps, order)

        # be careful at the order of phasespace_target and PS_regressed
        # order by default: H/thad/tlep/ISR
        PS_regressed, detjinv_regressed = Compute_ParticlesTensor.get_PS(HttISR_regressed, data_boost_reco)

        if sampling_Forward:
            flow_sample = self.flow(PS_regressed).sample()

            sample_mask_all = (flow_sample>=0) & (flow_sample<=1)
            sample_mask = torch.all(sample_mask_all, dim=1)

            flow_sample = flow_sample[sample_mask]
            PS_target_masked = PS_target[sample_mask]

            print(f"size flow_sample: {flow_sample.shape}")
            print(f"size PS_target: {PS_target_masked.shape}")
            
            #flow_loss = self.mmdLoss(flow_sample, PS_target_masked)

            # kernel: kernel type such as "multiscale" or "rbf"
            flow_loss = MMD(x=flow_sample, y=PS_target_masked, kernel='multiscale', device=device)
            print(flow_loss.shape)
        
        else:
            flow_loss = self.flow(PS_regressed).log_prob(PS_target)
        
        detjac = PS_rambo_detjacobian.log()
        return flow_loss, detjac, cond_X, PS_regressed

