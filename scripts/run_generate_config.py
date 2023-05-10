import sys
import subprocess
import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--maxFiles', type=int, default=-1, help='Maximum number of files created')
    parser.add_argument('--preTraining', action="store_true",  help='creates config files for pretraining')
    args = parser.parse_args()
        
    preTraining = args.preTraining
    input_dataset = args.input_dataset
    if preTraining:
        name = 'preTraining-MEMFlow'
    else:
        name = 'MEMFlow'
    description = ''
    numberJets = 15
    inputFeatures = 5
    numberLept = 1
    training_batchSizeTraining = 2048
    training_batchSizeValid = 2048
    nEpochs = 200
    training_sampleDim = 500000
    valid_sampleDim = 69993
    sampling_points = 100
    
    cond_outFeatures = [4]
    cond_hiddenFeatures = [32, 64, 128]
    cond_DimFeedForwardTransf = [512, 1024, 2048]
    cond_nHeadEncoder = [2]
    cond_noLayersEncoder = [3, 4, 5]
    cond_nHeadDecoder = [1, 2]
    cond_noLayersDecoder = [2, 3]
    cond_aggregate = [False]
    cond_noDecoders = [3]
    
    flow_nFeatures = [10, 20, 30]
    flow_nCond = 0 # this is set up by 'cond_outFeatures + 12 - flow_nFeatures'
    flow_nTransforms = [x + 4 for x in range(6)]
    flow_bins = [16, 32, 64, 128]
    flow_autoregressive = [True, False]
    flow_hidden_layers = [x + 3 for x in range(5)]
    flow_hidden_layersSize = [32, 64, 128, 256, 512]
    
    #learningRate = [x*0.000001 for x in range(20)]
    learningRate = [0.000001]

    cond_aggregate = ['--cond_aggregate' if x is True else '--no-cond_aggregate' for x in cond_aggregate]
    flow_autoregressive = ['--flow_autoregressive' if x is True else '--no-flow_autoregressive' for x in flow_autoregressive ]
    
    i = 0
    current_path = os.path.dirname(os.path.realpath(__file__))
    
    if preTraining:
        for cond_outFeatures_value in cond_outFeatures:
            for cond_hiddenFeatures_value in cond_hiddenFeatures:
                for cond_DimFeedForwardTransf_value in cond_DimFeedForwardTransf:
                    for cond_nHeadEncoder_value in cond_nHeadEncoder:
                        for cond_noLayersEncoder_value in cond_noLayersEncoder:
                            for cond_nHeadDecoder_value in cond_nHeadDecoder:
                                for cond_noLayersDecoder_value in cond_noLayersDecoder:
                                    for cond_noDecoders_value in cond_noDecoders:    
                                        for learningRate_value in learningRate:                                                                        
                                            version = 'v' + str(i)
                                            subprocess.call(["python",
                                                            f'{current_path}/../memflow/unfolding_flow/generate_config.py',
                                                            f"--input_dataset={input_dataset}",
                                                            f'--name={name}',
                                                            f'--version={version}',
                                                            f'--description={description}',
                                                            f'--numberJets={numberJets}',
                                                            f'--numberLept={numberLept}',
                                                            f'--inputFeatures={inputFeatures}',
                                                            f'--cond_outFeatures={cond_outFeatures_value}',
                                                            f'--cond_hiddenFeatures={cond_hiddenFeatures_value}',
                                                            f'--cond_DimFeedForwardTransf={cond_DimFeedForwardTransf_value}',
                                                            f'--cond_nHeadEncoder={cond_nHeadEncoder_value}',
                                                            f'--cond_noLayersEncoder={cond_noLayersEncoder_value}',
                                                            f'--cond_nHeadDecoder={cond_nHeadDecoder_value}',
                                                            f'--cond_noLayersDecoder={cond_noLayersDecoder_value}',
                                                            f'{cond_aggregate[0]}',
                                                            f'--cond_noDecoders={cond_noDecoders_value}',
                                                            f'--training_batchSizeTraining={training_batchSizeTraining}',
                                                            f'--training_batchSizeValid={training_batchSizeValid}',
                                                            f'--learningRate={learningRate_value}',
                                                            f'--nEpochs={nEpochs}',
                                                            f'--training_sampleDim={training_sampleDim}',
                                                            f'--valid_sampleDim={valid_sampleDim}',
                                                            f'--sampling_points={sampling_points}'])
                                                                                
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
                                                                            
                                                                    subprocess.call(["python", 
                                                                                    f'{current_path}/../memflow/unfolding_flow/generate_config.py',
                                                                                    f'--input_dataset={input_dataset}',
                                                                                    f'--name={name}',
                                                                                    f'--version={version}',
                                                                                    f'--description={description}',
                                                                                    f'--numberJets={numberJets}',
                                                                                    f'--numberLept={numberLept}',
                                                                                    f'--inputFeatures={inputFeatures}',
                                                                                    f'--cond_outFeatures={cond_outFeatures_value}',
                                                                                    f'--cond_hiddenFeatures={cond_hiddenFeatures_value}',
                                                                                    f'--cond_DimFeedForwardTransf={cond_DimFeedForwardTransf_value}',
                                                                                    f'--cond_nHeadEncoder={cond_nHeadEncoder_value}',
                                                                                    f'--cond_noLayersEncoder={cond_noLayersEncoder_value}',
                                                                                    f'--cond_nHeadDecoder={cond_nHeadDecoder_value}',
                                                                                    f'--cond_noLayersDecoder={cond_noLayersDecoder_value}',
                                                                                    f'{cond_aggregate[0]}',
                                                                                    f'--cond_noDecoders={cond_noDecoders_value}',
                                                                                    f'--flow_nFeatures={flow_nFeatures_value}',
                                                                                    f'--flow_nCond={cond_outFeatures_value + 12 - flow_nFeatures_value}',
                                                                                    f'--flow_nTransforms={flow_nTransforms_value}',
                                                                                    f'--flow_bins={flow_bins_value}',
                                                                                    f'{flow_autoregressive_value}',
                                                                                    f'--training_batchSizeTraining={training_batchSizeTraining}',
                                                                                    f'--training_batchSizeValid={training_batchSizeValid}',
                                                                                    f'--learningRate={learningRate_value}',
                                                                                    f'--nEpochs={nEpochs}',
                                                                                    f'--training_sampleDim={training_sampleDim}',
                                                                                    f'--valid_sampleDim={valid_sampleDim}',
                                                                                    f'--sampling_points={sampling_points}',
                                                                                    f'--flow_hidden_layers={flow_hidden_layers_value}',
                                                                                    f'--flow_hidden_layersSize={flow_hidden_layersSize_value}'])
                                                                            
                                                                    i = i+1
                                                                            
                                                                    if i >= args.maxFiles and args.maxFiles != -1:
                                                                        sys.exit()