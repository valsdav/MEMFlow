from omegaconf import OmegaConf
import sys
import argparse
import os

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--name', type=str, default='TEST', help='name of config (by default: \'TEST\')')
    parser.add_argument('--version', type=str, default='v0', help='version of config (by default: \'v0\')')
    parser.add_argument('--description', type=str, default='', help='description of config (by default: \'\')')
    parser.add_argument('--numberJets', type=int, default=15, help='number of jets in one event (by default: 15)')
    parser.add_argument('--numberLept', type=int, default=1, help='number of leptons in one event (by default: 1)')
    parser.add_argument('--inputFeatures', type=int, default=5, help='number of input features (by default: 5)')

    parser.add_argument('--flow_nFeatures', type=int, default=10, help='number of features in NF (by default: 10)')
    parser.add_argument('--flow_nCond', type=int, default=34, help='number of context features in NF (by default: 34)')
    parser.add_argument('--flow_nTransforms', type=int, default=5, help='number of transformations in NF (by default: 5)')
    parser.add_argument('--flow_bins', type=int, default=32, help='number of bins in NF (by default: 32)')
    parser.add_argument('--flow_autoregressive', action=argparse.BooleanOptionalAction, default=True, help='type of transformations in NF (by default: True - autoregressive)')
    parser.add_argument('--training_batchSizeTraining', type=int, default=2048,
                        help='batch size for training data (by default: 2048)')
    parser.add_argument('--training_batchSizeValid', type=int, default=2048,
                        help='batch size for validation data (by default: 2048)')
    parser.add_argument('--learningRate', type=float, default=1e-5, help='learning rate for training (by default: )')
    parser.add_argument('--nEpochs', type=int, default=5, help='number of epochs (by default: 1e-5)')
    parser.add_argument('--training_sampleDim', type=int, default=500000, help='dimension of training data (by default: 500000)')
    parser.add_argument('--valid_sampleDim', type=int, default=69993, help='dimension of validation data (by default: 69993)')
    parser.add_argument('--sampling_points', type=int, default=10, help='number of sampling points in NF (by default: 10)')

    parser.add_argument('--flow_hidden_layers', type=int, default=4, help='number of hidden layers in NF (by default: 4)')
    parser.add_argument('--flow_hidden_layersSize', type=int, default=128, help='size of hidden layers in NF (by default: 128)')

    parser.add_argument('--cond_outFeatures', type=int, default=4, help='No of output features of CondTransformer (by default: 4 - 4 momenta)')
    parser.add_argument('--cond_hiddenFeatures', type=int, default=32, help='No of hidden features in CondTransformer (by default: 32)')
    parser.add_argument('--cond_DimFeedForwardTransf', type=int, default=512, help='No. dimensions of FeedForward Transformer in CondTransformer (by default: 512)')
    parser.add_argument('--cond_nHeadEncoder', type=int, default=2, help='No. of Heads in multihead attention for Encoders in CondTransformer (by default: 2)')
    parser.add_argument('--cond_noLayersEncoder', type=int, default=3, help='No. of Encoder layers in CondTransformer (by default: 3)')
    parser.add_argument('--cond_nHeadDecoder', type=int, default=1, help='No. of Heads in multihead attention of Decoders in CondTransformer (by default: 1)')
    parser.add_argument('--cond_noLayersDecoder', type=int, default=1, help='No. of Decoder layers in CondTransformer (by default: 1)')
    parser.add_argument('--cond_aggregate', action=argparse.BooleanOptionalAction, default=False, help='set aggregate in CondTransformer (by default: False)')
    parser.add_argument('--cond_noDecoders', type=int, default=3, help='No. of Decoders in CondTransformer (by default: 3)')
    args = parser.parse_args()
    print(args)
        
    config = {
        "name": args.name,
        "version": args.version,
        "description": args.description,
        "input_dataset": args.input_dataset,
        "input_shape": {
            "number_jets": args.numberJets,
            "number_lept": args.numberLept,
            "input_features": args.inputFeatures
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
            "batch_size_training": args.training_batchSizeTraining,
            "batch_size_validation": args.training_batchSizeValid,
            "nepochs": args.nEpochs,
            "training_sample": args.training_sampleDim,
            "validation_sample": args.valid_sampleDim,
            "sampling_points": args.sampling_points
        },
        "conditioning_transformer":{
            "out_features": args.cond_outFeatures, # the 4 momenta
            "hidden_features": args.cond_hiddenFeatures,
            "dim_feedforward_transformer": args.cond_DimFeedForwardTransf,
            "nhead_encoder": args.cond_nHeadEncoder,
            "no_layers_encoder": args.cond_noLayersEncoder,
            "nhead_decoder": args.cond_nHeadDecoder,
            "no_layers_decoder": args.cond_noLayersDecoder,
            "aggregate": args.cond_aggregate,
            "no_decoders": args.cond_noDecoders
        }
    }

    conf = OmegaConf.create(config)
    
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Save configs in MEMFlow/scripts
    if(not os.path.exists(f'{current_path}/../../scripts/configs')):
        os.makedirs(f'{current_path}/../../scripts/configs')
        print("Create configs directory")
    
    with open(f"{current_path}/../../scripts/configs/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf))