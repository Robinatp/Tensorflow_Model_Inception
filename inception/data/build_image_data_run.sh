#!/bin/bash
## Note the locations of the train and validation data.
TRAIN_DIR=`pwd`/demo_picture/train/

VALIDATION_DIR=`pwd`/demo_picture/validation/
LABELS_FILE=`pwd`/demo_picture/labels.txt

# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=`pwd`/demo_picture/

python build_image_data.py \
  --train_directory=${TRAIN_DIR} \
  --validation_directory=${VALIDATION_DIR} \
  --output_directory=${OUTPUT_DIRECTORY} \
  --labels_file=${LABELS_FILE} \
  --train_shards=10 \
  --validation_shards=4 \
  --num_threads=2
