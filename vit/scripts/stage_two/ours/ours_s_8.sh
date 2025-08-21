#!/bin/bash
me=$(basename "$0")
mkdir -p ./runs/${me}

work_dir=./runs/${me}/2kpseudo_kd
mkdir -p $work_dir

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH TRAIN_CONFIG=$(dirname $0)/../tools/config/qat_syn.json python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=63667 \
    $(dirname "$0")/../tools/train_qat.py \
    --work_dir $work_dir \
    --pseudo_data INSERT_DATA_PATH_HERE \
    --enable_kd true \
    --kd_modules block \
    --config ./configs/mask_rcnn/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    --pretrained_path ./pretrained/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
    --original_loss_weight 1.0 \
    --kd_loss_weight 1.0 \
    --mse_loss_weight 1.0 \
    --launcher pytorch
