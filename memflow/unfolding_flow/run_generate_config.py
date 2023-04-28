import sys
import subprocess

if __name__ == '__main__':
    
    if (len(sys.argv) == 1):
        raise Exception('Need at least 1 CLI argument - path to dataset')
        
    path_to_dataset = sys.argv[1]
    name = 'test'
    description = ''
    numberJets = 15
    jetsFeatures = 5
    numberLept = 1
    leptonFeatures = 3
    training_batchSize = 2048
    training_batchSizeTraining = 2048
    training_batchSizeValid = 2048
    nEpochs = 50
    training_sampleDim = 500000
    valid_sampleDim = 69993
    sampling_points = 100
    
    cond_outFeatures = [32, 64, 128]
    cond_nHead = [1, 2]
    cond_noLayers = [x+3 for x in range(7)]
    
    flow_nFeatures = [10, 20, 30]
    flow_nCond = 0 # this is set up by 'cond_outFeatures + 12 - flow_nFeatures'
    flow_nTransforms = [x + 4 for x in range(6)]
    flow_bins = [16, 32, 64, 128]
    flow_autoregressive = [True, False]
    flow_hidden_layers = [x + 3 for x in range(5)]
    flow_hidden_layersSize = [32, 64, 128, 256, 512]
    
    learningRate = [x*0.000001 for x in range(20)]
    
    
    
    
    
    i = 0
    
    for cond_outFeatures_value in cond_outFeatures:
        for cond_nHead_value in cond_nHead:
            for cond_noLayers_value in cond_noLayers:
                for flow_nFeatures_value in flow_nFeatures:
                    for flow_nTransforms_value in flow_nTransforms:
                        for flow_bins_value in flow_bins:
                            for flow_autoregressive_value in flow_autoregressive:
                                for flow_hidden_layers_value in flow_hidden_layers:
                                    for flow_hidden_layersSize_value in flow_hidden_layersSize:
                                        for learningRate_value in learningRate:
                                            
                                            version = 'v' + str(i)
                                            i = i+1
                                            
                                            subprocess.call(["python", "./generate_config.py",
                                                             f'--path_to_dataset={path_to_dataset}',
                                                             f'--name={name}',
                                                             f'--version={version}',
                                                             f'--description={description}',
                                                             f'--numberJets={numberJets}',
                                                             f'--jetsFeatures={jetsFeatures}',
                                                             f'--numberLept={numberLept}',
                                                             f'--leptonFeatures={leptonFeatures}',
                                                             f'--cond_outFeatures={cond_outFeatures_value}',
                                                             f'--cond_nHead={cond_nHead_value}',
                                                             f'--cond_noLayers={cond_noLayers_value}',
                                                             f'--flow_nFeatures={flow_nFeatures_value}',
                                                             f'--flow_nCond={cond_outFeatures_value + 12 - flow_nFeatures_value}',
                                                             f'--flow_nTransforms={flow_nTransforms_value}',
                                                             f'--flow_bins={flow_bins_value}',
                                                             f'--flow_autoregressive={flow_autoregressive_value}',
                                                             f'--training_batchSizeTraining={training_batchSizeTraining}',
                                                             f'--training_batchSizeValid={training_batchSizeValid}',
                                                             f'--learningRate={learningRate_value}',
                                                             f'--nEpochs={nEpochs}',
                                                             f'--training_sampleDim={training_sampleDim}',
                                                             f'--valid_sampleDim={valid_sampleDim}',
                                                             f'--sampling_points={sampling_points}',
                                                             f'--flow_hidden_layers={flow_hidden_layers_value}',
                                                             f'--flow_hidden_layersSize={flow_hidden_layersSize_value}'])
    
    
    
    
    