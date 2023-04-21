import torch.nn as nn
import torch
import numpy as np


class ConditioningTransformerLayer(nn.Module):
    def __init__(self, no_jets, jets_features, no_lept, lepton_features, out_features, nhead, no_layers, dtype=torch.float32):
        super().__init__()
    
        self.no_jets = no_jets
        self.no_lept = no_lept
        self.lin_jet = nn.Linear(in_features=jets_features,
                                 out_features=out_features - 1, dtype=dtype)
        self.lin_lept = nn.Linear(in_features=lepton_features,
                                  out_features=out_features - 1, dtype=dtype)
        self.lin_met = nn.Linear(in_features=3,
                                 out_features=out_features - 1, dtype=dtype)
        self.lin_boost = nn.Linear(in_features=4,
                                   out_features=out_features - 1, dtype=dtype)

        self.gelu = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_features,
                                                   nhead=nhead,
                                                   batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=no_layers)

        self.register_buffer('ones', torch.ones(no_jets, 1))
        self.register_buffer('two', 2*torch.ones(no_lept, 1))
        self.register_buffer('three', 3*torch.ones(1, 1))
        self.register_buffer('four', 4*torch.ones(1, 1))
        

    def reset_parameters(self):
        self.lin_jet.reset_parameters()
        self.lin_lept.reset_parameters()
        self.lin_met.reset_parameters()
        self.lin_boost.reset_parameters()

    def forward(self, batch_jet, batch_lepton, batch_met, batch_boost, mask_jets, mask_lepton, mask_met, mask_boost):

        jets_afterLin = self.gelu(self.lin_jet(
            batch_jet) * mask_jets[:, :, None])
        lept_afterLin = self.gelu(self.lin_lept(batch_lepton))
        met_afterLin = self.gelu(self.lin_met(batch_met))
        boost_afterLin = self.gelu(self.lin_boost(batch_boost))

        batch_size = batch_jet.size(0)       

        jet_afterLin_andLabel = torch.cat((jets_afterLin, 
                                           self.ones.expand(batch_size, *list(self.ones.shape))), dim=-1)
        lept_afterLin_andLabel = torch.cat((lept_afterLin, 
                                            self.two.expand(batch_size, *list(self.two.shape))), dim=-1)
        met_afterLin_andLabel = torch.cat((met_afterLin, 
                                           self.three.expand(batch_size, *list(self.three.shape))), dim=-1)
        boost_afterLin_andLabel = torch.cat((boost_afterLin, 
                                             self.four.expand(batch_size, *list(self.four.shape))), dim=-1)

        transformer_input = torch.concat(
            (boost_afterLin_andLabel, lept_afterLin_andLabel, met_afterLin_andLabel, jet_afterLin_andLabel), dim=1)
        transformer_mask = torch.concat(
            (mask_boost, mask_lepton, mask_met, mask_jets), dim=1)

        transformer_output = self.transformer_encoder(
            transformer_input, src_key_padding_mask=transformer_mask == 0)

        # `computing ther average of not masked objects`
        transformer_output_sum = torch.sum(
            transformer_output * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]
        
        N_valid_objects = torch.sum(transformer_mask, dim=1).unsqueeze(1)  #[B, 1]

        conditional_input = transformer_output_sum / N_valid_objects

        x1 = (batch_boost[:, :, 3] + batch_boost[:, :, 2]) / 13000.
        x2 = (batch_boost[:, :, 3] - batch_boost[:, :, 2]) / 13000.
        
        return torch.cat((x1, x2,conditional_input), dim=1)
