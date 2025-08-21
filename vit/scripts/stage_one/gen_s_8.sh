#!/bin/bash
me=$(basename "$0")

work_dir="./runs/${me}/swinS"
mkdir -p $work_dir

for ((i = 0; i < 8; i++)); do
    GENERATION_CONFIG=./tools/config/generation.json \
        python -m tools.generation \
        --config ./configs/mask_rcnn/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
        --dataset_config ./configs/datasets/generation.py \
        --pretrained_weights ./pretrained/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
        --work_dir $work_dir \
        --calibration_size 256 \
        --batch_size 12 \
        --devices $i \
        | tee ${work_dir}/g_${i}.log &
done

wait
