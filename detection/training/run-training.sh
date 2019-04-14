#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/tensorflow/models:/tensorflow/models/slim

python3 object_detection/train.py --logtostderr \
    --train_dir=./models/train \
    --pipeline_config_path=/training/faster_rnn_resnet101.config