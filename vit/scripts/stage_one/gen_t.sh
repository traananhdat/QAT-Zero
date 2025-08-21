#!/bin/bash

me=$(basename "$0")

work_dir="./runs/${me}/swinT"
mkdir -p $work_dir

GENERATION_CONFIG=./tools/config/generation.json \
        python -m tools.generation \
        --config ./configs/mask_rcnn/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
        --dataset_config ./configs/datasets/generation.py \
        --pretrained_weights ./pretrained/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth \
        --work_dir $work_dir \
        --calibration_size 2048 \
        --batch_size 12 \
        | tee ${work_dir}/g_${i}.log
