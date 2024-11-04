#!/bin/bash

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

cd $1
pip install -e .

python scripts/run_transferFlow_paperVersion_pretrained_v3_onlyExist-noMDMM_MET_OnlyPartons_btag.py \
      --path-config  "$1/$2" --output-dir  $3 --on-GPU $4