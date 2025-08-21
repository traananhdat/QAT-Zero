#!/bin/bash

outdir=$(pwd)/data/COCO2017/cocoonebox

echo "using image in $outdir"

config_file="$(pwd)/data/cocoonebox.yaml"
escaped_outdir=$(printf '%s\n' "$outdir" | sed 's/[\/&]/\\&/g')

sed -i "s/PLACEHOLDER/$escaped_outdir/" "$config_file"

GENERATION_CONFIG=./generation/config/genlabel.json python -m generation.main \
    --teacher_weights yolo11s.pt \
    --relabel_weights yolo11s.pt

# rollback to the original state.
sed -i "s/$escaped_outdir/PLACEHOLDER" "$config_file"
