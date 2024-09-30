#!/usr/bin/env bash

#SBATCH --account=jinm11
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:00

### Setup ###

DATA_PATH=/p/data1/pli/Projects/Volume_Reconstruction/Vervet1818_VisualCortex/2019/PLI-PM/Transmittance_Retardation/nifti/
OUT_PATH=/p/scratch/cjinm11/Private/oberstrass1/tmp/features/
CKPT=models/cl-3d/resnet50_planes8_962-1083_sphere_small.ckpt

echo "Starting script"
source $HOME/.bashrc

echo "Enabling environment..."
conda activate cl-3d

export CUDA_VISIBLE_DEVICES=0,1,2,3

export HDF5_USE_FILE_LOCKING='FALSE'

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i
export MASTER_PORT=23519


### Run script ###

srun -c 16 python scripts/apply-encoder.py \
  --ckpt=$CKPT \
  --trans="$DATA_PATH/*NTransmittance*" \
  --dir="$DATA_PATH/*Direction*" \
  --ret="$DATA_PATH/*Retardation*" \
  --out=$OUT_PATH \
  --num_workers=16 \
  --batch_size=16 \
  --overlap=0.5 \
  --ram \
  --dtype=float16 \

