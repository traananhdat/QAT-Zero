#!/bin/bash

outdir=$(pwd)/data/COCO2017/cocoonebox

echo "generating image to $outdir"

python -m label.generate_one_label \
    --numImages 5120 \
    --outdir $outdir
