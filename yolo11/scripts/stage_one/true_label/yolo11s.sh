#!/bin/bash

GENERATION_CONFIG=./generation/config/true_label.json \
    python -m generation.main \
    --device cuda:0 \
    --teacher_weights yolo11s.pt \
    --relabel_weights yolo11s.pt
