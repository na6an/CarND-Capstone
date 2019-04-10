#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/tensorflow/models:/tensorflow/models/slim

python3 /training/tf_record_gen.py \
    --data_dir=/data \
    --output_path=/output/data_tf.record \
    --label_map_path=/training/tl_label_map.pbtxt
