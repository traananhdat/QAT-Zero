#!/bin/bash
me=$(basename "$0")

mkdir -p ./runs/${me}

work_dir=./runs/${me}/2kreal

mkdir -p $work_dir

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH TRAIN_CONFIG=$(dirname $0)/../tools/config/2048_qat.json python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=63667 \
    $(dirname "$0")/../tools/train_qat.py \
    --work_dir $work_dir \
    --config ./configs/mask_rcnn/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    --pretrained_path ./pretrained/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
    --launcher pytorch
