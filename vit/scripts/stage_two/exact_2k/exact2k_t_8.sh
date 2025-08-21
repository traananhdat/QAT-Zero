#!/bin/bash
me=$(basename "$0")

mkdir -p ./runs/${me}

work_dir=./runs/${me}/2kreal

mkdir -p $work_dir

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH TRAIN_CONFIG=$(dirname $0)/../tools/config/2048_qat.json python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=63667 \
    $(dirname "$0")/../tools/train_qat.py \
    --work_dir $work_dir \
    --launcher pytorch
