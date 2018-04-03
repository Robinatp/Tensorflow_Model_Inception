# Build the TFRecords version of the ImageNet data.
# convert the ILSVRC2012 imagenet dataset to tfrecord,
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-01023-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128

WORK_DIR=/workspace/zhangbin/master/models/research/inception/inception
BUILD_SCRIPT="${WORK_DIR}/data/build_imagenet_data.py"

TRAIN_DIRECTORY=/workspace/zhangbin/dataset_robin/imagenet-data/raw-data/ILSVRC2012_img_train/
VALIDATION_DIRECTORY=/workspace/zhangbin/dataset_robin/imagenet-data/raw-data/ILSVRC2012_img_val/

OUTPUT_DIRECTORY=/workspace/zhangbin/dataset_robin/imagenet-data/raw-data/tfrecod/
IMAGENET_METADATA_FILE="${WORK_DIR}/data/imagenet_metadata.txt"
LABELS_FILE="${WORK_DIR}/data/imagenet_lsvrc_2015_synsets.txt"
BOUNDING_BOX_FILE=/workspace/zhangbin/dataset_robin/imagenet-data/raw-data/imagenet_2012_bounding_boxes.csv

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}" \
  --validation_shards=64
