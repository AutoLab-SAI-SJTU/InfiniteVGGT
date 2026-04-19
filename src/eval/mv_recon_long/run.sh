#!/bin/bash

set -e

workdir='..'  
model_name='StreamVGGT'
ckpt_name='checkpoints' 

model_weights="/Your/model_weights/path"
input_dir="/Your/input/path" 
gt_path="/Your/gt/path"

output_dir="/Your/out_put/path"

# ==========================================

echo "========================================"
echo "Running Evaluation"
echo "Weights: ${model_weights}"
echo "Input:   ${input_dir}"
echo "GT:      ${gt_path}"
echo "Output:  ${output_dir}"
echo "========================================"

accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon_long/launch.py \
    --weights "$model_weights" \
    --input_dir "$input_dir" \
    --gt_path "$gt_path" \
    --output_dir "$output_dir" \
    --skip_inference # if you already had the predicion results

echo "Done. Results saved in ${output_dir}"