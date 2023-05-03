import argparse
import pathlib

import os
from os import listdir
from os.path import isfile, join

# get absolute paths of Files from directory
def absoluteFilePaths(directory):
    paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            paths.append(os.path.abspath(os.path.join(dirpath, f)))
            
    return paths

# make a directory (dir) if it doesn't exist
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_directory', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--env_path', type=str, default="-1", help='path to python environment')
    parser.add_argument('--on_CPU', type=bool, default=0, help='run on CPU boolean, by default: 0')
    args = parser.parse_args()
    
    conf_dir = args.config_directory
    env_path = args.env_path
    on_CPU = args.on_CPU # by default run on GPU
    
    # get path of 'run_model' script
    scriptPath = str(pathlib.Path(__file__).parent.resolve()) + "/run_model.py"
    
    # get absolute paths of all config files in conf_dir
    confFiles = absoluteFilePaths(conf_dir)
    
    # if environment path is passed, get its absolute path
    if (env_path != "-1"):
        abs_envPath = os.path.abspath(env_path)
        
    model_results = f"{os.getcwd()}/logs"
    
    mkdir_p(model_results)
    
    job_directory = f"{os.getcwd()}/jobs"
    
    # create job directory (where job config files are kept)
    mkdir_p(job_directory)
        
        
    for i, conf_path in enumerate(confFiles):
        
        # one job file for each config file
        job_file = os.path.join(job_directory, f"{i}.job")

        with open(job_file, 'w') as fh:
            
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name=training_{i}.job\n")
            fh.writelines("#SBATCH --account=gpu_gres\n")
            fh.writelines("#SBATCH --partition=gpu\n")
            fh.writelines("#SBATCH --nodes=1        # request to run job on single node\n")
            fh.writelines("#SBATCH --ntasks=5       # request 5 CPU's\n")
            fh.writelines("#SBATCH --gres=gpu:1     # request 1 GPU's on machine\n")
            fh.writelines("#SBATCH --mem=4000M      # memory (per job)\n")
            fh.writelines("#SBATCH --time=0-00:30   # time  in format DD-HH:MM\n")    
            
            if (env_path != "-1"):
                fh.writelines("\n\n")
                fh.writelines("# Activate environment:\n")
                fh.writelines(f"source {abs_envPath}\n")
            
            fh.writelines("\n\n")
            fh.writelines("# each node has local /scratch space to be used during job run\n")
            fh.writelines("mkdir -p /scratch/$USER/${SLURM_JOB_ID}\n")
            fh.writelines("export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}\n")
            
            fh.writelines("\n\n")
            fh.writelines("echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES\n")
            fh.writelines("# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)\n")
            fh.writelines(f"python {scriptPath} --path_config={conf_path} --on_CPU={on_CPU}\n")
            
            fh.writelines("\n\n")
            fh.writelines("# cleaning of temporal working dir when job was completed:\n")
            fh.writelines("rmdir  -rf /scratch/$USER/${SLURM_JOB_ID}\n")
            

        # os.system(f"sbatch {job_file}")
                
        
        
        
        
        
        
        