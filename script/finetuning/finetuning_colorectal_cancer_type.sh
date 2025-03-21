#!/bin/bash
#SBATCH --partition=ai4bio
#SBATCH --nodes=1
#SBATCH --ntasks=4           # 4 tasks * 4 nodes = 16
#SBATCH --ntasks-per-node=4   # 4 tasks on each node
#SBATCH --gres=gpu:4          # 4 GPUs per node
#SBATCH --chdir=/mnt/petrelfs/zhengpeng/scWGBS-GPT
export CFLAGS="-std=c99"
export OMP_NUM_THREADS=1
TRAINING_ARGS_PATH="config/finetuning/colorectal_cancer_type/training_args_fp16.json"

# If you have 1 nodes each with 4 GPUs, that's 4 GPUs total.
# You can still do torchrun with --nproc_per_node=4 and specify --nnodes=1.
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=2156 \
  finetuning/finetuning.py --training_args_path "$TRAINING_ARGS_PATH"