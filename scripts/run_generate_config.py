import sys
import subprocess
import argparse
import os
from omegaconf import OmegaConf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--maxFiles', type=int, default=-1, help='Maximum number of files created')
    parser.add_argument('--preTraining', action="store_true",  help='creates config files for pretraining')
    args = parser.parse_args()
        
    preTraining = args.preTraining
    input_dataset = args.input_dataset
    if preTraining:
        name = 'preTraining-MEMFlow_noprov'
    else:
        name = 'MEMFlow'
    description = ''
    numberJets = 15
    inputFeatures = 5
    numberLept = 1
    training_batchSizeTraining = 2048
    training_batchSizeValid = 2048
    nEpochs = 1000
    training_sampleDim = 500000
    valid_sampleDim = 69993
    sampling_points = 100
    nEpochsPatience = 20
    
    cond_outFeatures = [4]
    cond_hiddenFeatures = [16, 32, 64]
    cond_DimFeedForwardTransf = [1024, 2048]
    cond_nHeadEncoder = [1, 4, 8]
    cond_noLayersEncoder = [1, 2, 3, 4]
    cond_nHeadDecoder = [1, 4, 8]
    cond_noLayersDecoder = [1, 2, 3, 4]
    cond_aggregate = False
    cond_noDecoders = [4]
    
    flow_nFeatures = [10, 20, 30]
    flow_nCond = 0 # this is set up by 'cond_outFeatures + 12 - flow_nFeatures'
    flow_nTransforms = [x + 4 for x in range(6)]
    flow_bins = [16, 32, 64, 128]
    flow_autoregressive = [True, False]
    flow_hidden_layers = [x + 3 for x in range(5)]
    flow_hidden_layersSize = [32, 64, 128, 256, 512]
    
    #learningRate = [x*0.000001 for x in range(20)]
    learningRate = [1e-3]
    
    i = 0
    current_path = os.path.dirname(os.path.realpath(__file__))
    
    if preTraining:
        for cond_outFeatures_value in cond_outFeatures:
            for k, cond_hiddenFeatures_value in enumerate(cond_hiddenFeatures):
                for cond_DimFeedForwardTransf_value in cond_DimFeedForwardTransf:
                    for cond_nHeadEncoder_value in [-1]:
                        for j, cond_noLayersEncoder_value in enumerate(cond_noLayersEncoder):
                            for cond_nHeadDecoder_value in [-1]:
                                for cond_noLayersDecoder_value in [-2]:
                                    for cond_noDecoders_value in cond_noDecoders:    
                                        for learningRate_value in learningRate:                                                                        
                                            version = 'v' + str(i)      

                                            config = {
                                                    "name": name + f'_{version}',
                                                    "version": version,
                                                    "description": description,
                                                    "input_dataset": input_dataset,
                                                    "input_shape": {
                                                        "number_jets": numberJets,
                                                        "number_lept": numberLept,
                                                        "input_features": inputFeatures
                                                    },
                                                    "training_params": {
                                                        "lr": learningRate_value,
                                                        "batch_size_training": training_batchSizeTraining,
                                                        "batch_size_validation": training_batchSizeValid,
                                                        "nepochs": nEpochs,
                                                        "training_sample": training_sampleDim,
                                                        "validation_sample": valid_sampleDim,
                                                        "sampling_points": sampling_points,
                                                        "nEpochsPatience": nEpochsPatience
                                                    },
                                                    "conditioning_transformer":{
                                                        "out_features": cond_outFeatures_value, # the 4 momenta
                                                        "hidden_features": cond_hiddenFeatures_value,
                                                        "dim_feedforward_transformer": cond_DimFeedForwardTransf_value,
                                                        "nhead_encoder": cond_nHeadEncoder[k],
                                                        "no_layers_encoder": cond_noLayersEncoder_value,
                                                        "nhead_decoder": cond_nHeadDecoder[k],
                                                        "no_layers_decoder": cond_noLayersDecoder[j],
                                                        "aggregate": cond_aggregate,
                                                        "no_decoders": cond_noDecoders_value
                                                    }
                                                }

                                            conf = OmegaConf.create(config)
                                            outputDir = os.path.abspath(args.output_dir)
                                            # Save configs in MEMFlow/scripts
                                            if(not os.path.exists(f'{outputDir}')):
                                                os.makedirs(f'{outputDir}')
                                                print("Create configs directory")
                                            
                                            with open(f"{outputDir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
                                                fo.write(OmegaConf.to_yaml(conf))
                                                                                
                                            i = i+1
                                                                                
                                            if i >= args.maxFiles and args.maxFiles != -1:
                                                sys.exit()
    
    else:
        for cond_outFeatures_value in cond_outFeatures:
            for cond_hiddenFeatures_value in cond_hiddenFeatures:
                for cond_DimFeedForwardTransf_value in cond_DimFeedForwardTransf:
                    for cond_nHeadEncoder_value in cond_nHeadEncoder:
                        for cond_noLayersEncoder_value in cond_noLayersEncoder:
                            for cond_nHeadDecoder_value in cond_nHeadDecoder:
                                for cond_noLayersDecoder_value in cond_noLayersDecoder:
                                    for cond_noDecoders_value in cond_noDecoders:
                                        for flow_nFeatures_value in flow_nFeatures:
                                            for flow_nTransforms_value in flow_nTransforms:
                                                for flow_bins_value in flow_bins:
                                                    for flow_autoregressive_value in flow_autoregressive:
                                                        for flow_hidden_layers_value in flow_hidden_layers:
                                                            for flow_hidden_layersSize_value in flow_hidden_layersSize:
                                                                for learningRate_value in learningRate:
                                                                            
                                                                    version = 'v' + str(i)

                                                                    config = {
                                                                            "name": name + f'_{version}',
                                                                            "version": version,
                                                                            "description": description,
                                                                            "input_dataset": input_dataset,
                                                                            "input_shape": {
                                                                                "number_jets": numberJets,
                                                                                "number_lept": numberLept,
                                                                                "input_features": inputFeatures
                                                                            },
                                                                            "training_params": {
                                                                                "lr": learningRate_value,
                                                                                "batch_size_training": training_batchSizeTraining,
                                                                                "batch_size_validation": training_batchSizeValid,
                                                                                "nepochs": nEpochs,
                                                                                "training_sample": training_sampleDim,
                                                                                "validation_sample": valid_sampleDim,
                                                                                "sampling_points": sampling_points,
                                                                                "nEpochsPatience": nEpochsPatience
                                                                            },
                                                                            "conditioning_transformer":{
                                                                                "out_features": cond_outFeatures_value, # the 4 momenta
                                                                                "hidden_features": cond_hiddenFeatures_value,
                                                                                "dim_feedforward_transformer": cond_DimFeedForwardTransf_value,
                                                                                "nhead_encoder": cond_nHeadEncoder_value,
                                                                                "no_layers_encoder": cond_noLayersEncoder_value,
                                                                                "nhead_decoder": cond_nHeadDecoder_value,
                                                                                "no_layers_decoder": cond_noLayersDecoder_value,
                                                                                "aggregate": cond_aggregate,
                                                                                "no_decoders": cond_noDecoders_value
                                                                            },
                                                                            "unfolding_flow":{
                                                                                "nfeatures": flow_nFeatures_value,
                                                                                "ncond": cond_outFeatures_value + 12 - flow_nFeatures_value,
                                                                                "ntransforms": flow_nTransforms_value,
                                                                                "hidden_mlp": [flow_hidden_layersSize_value]*flow_hidden_layers_value,
                                                                                "bins": flow_bins_value,
                                                                                "autoregressive": flow_autoregressive_value
                                                                            }
                                                                        }

                                                                    conf = OmegaConf.create(config)
                                                                    outputDir = os.path.abspath(args.output_dir)
                                                                    # Save configs in MEMFlow/scripts
                                                                    if(not os.path.exists(f'{outputDir}')):
                                                                        os.makedirs(f'{outputDir}')
                                                                        print("Create configs directory")
                                                                    
                                                                    with open(f"{outputDir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
                                                                        fo.write(OmegaConf.to_yaml(conf))      
                                                                    
                                                                    i = i+1
                                                                            
                                                                    if i >= args.maxFiles and args.maxFiles != -1:
                                                                        sys.exit()