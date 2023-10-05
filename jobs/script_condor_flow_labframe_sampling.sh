#!/bin/bash

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

cd /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow
pip install -e .

       
python scripts/run_model_labframe_sampling.py  \
       --path-config  /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow/configs/$1 \
       --output-dir  /eos/user/d/dvalsecc/www/ttHbbAnalysis/MEMFlow/models_archive/$2 \
       --on-GPU --distributed
