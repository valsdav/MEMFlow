import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
import zuko
from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from memflow.unfolding_flow.utils import Compute_ParticlesTensor


class UnfoldingFlow(nn.Module):
    def __init__(self, model_path, log_mean, log_std, no_jets, no_lept, input_features, cond_hiddenFeatures=64,
                cond_dimFeedForward=512, cond_outFeatures=32, cond_nheadEncoder=4, cond_NoLayersEncoder=2,
                cond_nheadDecoder=4, cond_NoLayersDecoder=2, cond_NoDecoders=3, cond_aggregate=False,
                flow_nfeatures=12, flow_ncond=34, flow_ntransforms=5, flow_hiddenMLP_NoLayers=16,
                flow_hiddenMLP_LayerDim=128, flow_bins=16, flow_autoregressive=True, device=torch.device('cpu'), dtype=torch.float64):

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
                              base=BoxUniform,
                              base_args=[torch.ones(flow_nfeatures)*(-1.1),torch.ones(flow_nfeatures)*1.1], 
                              univariate_kwargs={"bound": 1.1 }, # Keeping the flow in the [-1.1,1.1] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.flow.transforms.insert(0, SimpleAffineTransform(0*torch.ones(flow_nfeatures),1*torch.ones(flow_nfeatures),
                                                     -1*torch.ones(flow_nfeatures), 1*torch.ones(flow_nfeatures)))
        
        
    def forward(self, data, device, noProv):

        (phasespace_intermediateParticles,
        phasespace_rambo_detjacobian,
        logScaled_reco, mask_lepton_reco, 
        mask_jets, mask_met, 
        mask_boost_reco, data_boost_reco) = data

        mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

        if (noProv):
            logScaled_reco = logScaled_reco[:,:,:-1]
        
        with torch.no_grad():
            cond_X = self.cond_transformer(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)
            HttISR_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR(cond_X, self.log_mean, self.log_std, device)
            PS_regressed, detjinv_regressed = Compute_ParticlesTensor.get_PS(HttISR_regressed, boost_regressed)

        #torch.clamp(phasespace_intermediateParticles, min=1e-2, max=1-1e-2, out=phasespace_intermediateParticles)

        flow_result = self.flow(PS_regressed).log_prob(phasespace_intermediateParticles)

        #inf_mask = torch.isinf(flow_result)
        #nonzeros = torch.count_nonzero(inf_mask)
        #print(f'inf:{nonzeros}\n')

        detjac = phasespace_rambo_detjacobian.log()

        return flow_result, detjac, cond_X, PS_regressed

