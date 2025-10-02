#!/bin/bash

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=6888 \
  finetuning/finetuning.py \
  --training_args_path config/finetuning/colorectal_cancer_type/training_args_fp16.json