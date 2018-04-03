bazel build //inception:flowers_train
INCEPTION_MODEL_DIR=`pwd`
# Path to the downloaded Inception-v3 model.
MODEL_PATH=`pwd`/tmp/checkpoints/inception-v3/model.ckpt-157585

# Directory where the flowers data resides.
FLOWERS_DATA_DIR=`pwd`/tmp/flowers-data/

# Directory where to save the checkpoint and events files.
TRAIN_DIR=`pwd`/tmp/flowers_train/

# Run the fine-tuning on the flowers data set starting from the pre-trained
# Imagenet-v3 model.
bazel-bin/inception/flowers_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --max_steps=10000 \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1

