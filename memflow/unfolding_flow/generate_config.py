#! /afs/cern.ch/user/a/adpetre/public/myenv/bin python

from omegaconf import OmegaConf
import sys
import argparse

if __name__ == '__main__':
    
    
    if (len(sys.argv) == 1):
        raise Exception('Need at least 1 CLI argument - path to dataset')
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--name', type=str, default='TEST', help='name of config')
    parser.add_argument('--version', type=str, default='v0', help='version of config')
    parser.add_argument('--description', type=str, default='', help='description of config')
    parser.add_argument('--numberJets', type=int, default=15, help='number of jets in one event')
    parser.add_argument('--jetsFeatures', type=int, default=5, help='number of jets features')
    parser.add_argument('--numberLept', type=int, default=1, help='number of leptons in one event')
    parser.add_argument('--leptonFeatures', type=int, default=3, help='number of lepton features')
    parser.add_argument('--cond_outFeatures', type=int, default=32, help='size of output of conditional transformer')
    parser.add_argument('--cond_nHead', type=int, default=1, 
                        help='number of heads in multihead attention of conditional transformer')
    parser.add_argument('--cond_noLayers', type=int, default=3, help='number of layers in conditional transformer')
    parser.add_argument('--flow_nFeatures', type=int, default=10, help='number of features in NF')
    parser.add_argument('--flow_nCond', type=int, default=34, help='number of context features in NF')
    parser.add_argument('--flow_nTransforms', type=int, default=5, help='number of transformations in NF')
    parser.add_argument('--flow_bins', type=int, default=32, help='number of bins in NF')
    parser.add_argument('--flow_autoregressive', type=bool, default=True, help='type of transformations in NF (by default autoregressive)')
    parser.add_argument('--training_batchSize', type=int, default=2048, help='batch size')
    parser.add_argument('--training_batchSizeTraining', type=int, default=2048, help='batch size for training data')
    parser.add_argument('--training_batchSizeValid', type=int, default=2048, help='batch size for validation data')
    parser.add_argument('--learningRate', type=float, default=1e-5, help='learning rate for training')
    parser.add_argument('--nEpochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--training_sampleDim', type=int, default=500000, help='dimension of training data')
    parser.add_argument('--valid_sampleDim', type=int, default=69993, help='dimension of validation data')
    parser.add_argument('--sampling_points', type=int, default=10, help='number of sampling points in NF')

    parser.add_argument('--flow_hidden_layers', type=int, default=4, help='number of hidden layers in NF')
    parser.add_argument('--flow_hidden_layersSize', type=int, default=128, help='size of hidden layers in NF')
    args = parser.parse_args()
    print(args)
        
    config = {
        "name": args.name,
        "version": args.version,
        "description": args.description,
        "input_dataset": args.path_to_dataset,
        "input_shape": {
            "number_jets": args.numberJets,
            "jets_features": args.jetsFeatures,
            "number_lept": args.numberLept,
            "lepton_features": args.leptonFeatures
        },
        "conditioning_transformer":{
            "out_features": args.cond_outFeatures,
            "nhead": args.cond_nHead,
            "no_layers": args.cond_noLayers
        },
        "unfolding_flow":{
            "nfeatures": args.flow_nFeatures,
            "ncond": args.flow_nCond,
            "ntransforms": args.flow_nTransforms,
            "hidden_mlp": [args.flow_hidden_layersSize]*args.flow_hidden_layers,
            "bins": args.flow_bins,
            "autoregressive": args.flow_autoregressive
        },
        "training_params":
        {
            "lr": args.learningRate,
            "batch_size": args.training_batchSize,
            "batch_size_training": args.training_batchSizeTraining,
            "batch_size_validation": args.training_batchSizeValid,
            "nepochs": args.nEpochs,
            "traning_sample": args.training_sampleDim,
            "validation_sample": args.valid_sampleDim,
            "sampling_points": args.sampling_points
        }
    }

    conf = OmegaConf.create(config)

    with open(f"configs/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf))