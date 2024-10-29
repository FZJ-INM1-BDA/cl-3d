#!/usr/bin/env bash

#SBATCH --account=jinm11
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:40:00

echo "Starting script"
source $HOME/.bashrc

echo "Enabling environment..."
conda activate cl-3d

export CUDA_VISIBLE_DEVICES=0,1,2,3

export HDF5_USE_FILE_LOCKING='FALSE'

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i
export MASTER_PORT=23519

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

### Run script ###

srun -c 64 python scripts/train.py \
  experiment=scaling/node_scaling \
  name=resnet50_4_nodes_512 \
  trainer.num_nodes=4 \
  debug=profiler
