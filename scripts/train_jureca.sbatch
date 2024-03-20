#!/usr/bin/env bash

#SBATCH --account=jinm11
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=4:00:00

echo "Starting script"
source $HOME/.bashrc

echo "Enabling environment..."
conda activate cl-3d

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

export HDF5_USE_FILE_LOCKING='FALSE'

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i
export MASTER_PORT=23519

### Define storage paths ###

# export DATA_DIR=datasets/
# export OUT_DIR=/dev/shm/$USER

# mkdir -p $OUT_DIR/vervet1818-3d-pairs/data/
# mkdir -p $OUT_DIR/vervet1818-3d/data/aa/transformation

# cp -rL $DATA_DIR/vervet1818-3d/data/aa/transformation $OUT_DIR/vervet1818-3d/data/aa/
# cp -L $DATA_DIR/vervet1818-3d-pairs/data/train_stack_aa_samples.csv $OUT_DIR/vervet1818-3d-pairs/data/
# cp -L $DATA_DIR/vervet1818-3d-pairs/data/train_aa_962-1083_chunk256_float16.h5 $OUT_DIR/vervet1818-3d-pairs/data/

# srun -c 32 python scripts/train.py \
#   experiment=$1 \
#   data_dir=$OUT_DIR \

### Run script ###

echo Run experiment $1

srun -c 32 python scripts/train.py \
  experiment=$1 \
  trainer.num_nodes=4 \
