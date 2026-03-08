#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_MODULE_LOADING=LAZY
conda activate env_isaaclab
cd /home/tosin/Documents/GitHub/IsaacLab
python3 scripts/reinforcement_learning/skrl/train_gru_kl_adaptive.py
