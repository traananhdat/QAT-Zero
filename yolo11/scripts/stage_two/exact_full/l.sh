#!/bin/bash

WBITS=(8 4 4)
ABITS=(8 4 8)

for ((i = 0; i < ${#WBITS[@]}; i++)); do
    model="yolo11l.pt"
    wbit="${WBITS[$i]}"
    abit="${ABITS[$i]}"
    echo "running $model exact in W$wbit A$abit"
    TRAIN_CONFIG="./external/config/qat_full.json" python -m external.yolo_main \
        --model "$model" \
        --model_quantize_mode.weight_bits "$wbit" \
        --model_quantize_mode.activation_bits "$abit"
done
