#!/bin/bash

output_dir=../ckpt
mkdir -p ${output_dir}/delta_ckpt/deepresonance/7b_tiva_v0

deepspeed --master_addr localhost --master_port 28459 train.py \
    --model deepresonance --prellmfusion --imagebind_embs_seq \
    --stage 1 \
    --server local \
    --save_path  ${output_dir}/delta_ckpt/deepresonance/7b_tiva_v0 \
    --log_path ${output_dir}/delta_ckpt/deepresonance/7b_tiva_v0/log \
    > ${output_dir}/stage_1.log 2>&1

deepspeed --master_addr localhost --master_port 28459 train.py \
    --model deepresonance --prellmfusion --imagebind_embs_seq \
    --stage 2 \
    --server local \
    --save_path  ${output_dir}/delta_ckpt/deepresonance/7b_tiva_v0 \
    --log_path ${output_dir}/delta_ckpt/deepresonance/7b_tiva_v0/log \
    > ${output_dir}/stage_2.log 2>&1

