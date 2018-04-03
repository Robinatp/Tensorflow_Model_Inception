
cation of where to place the flowers data
FLOWERS_DATA_DIR=`pwd`/tmp/flowers-data/
echo $FLOWERS_DATA_DIR

# build the preprocessing script.
cd tensorflow-models/inception
bazel build //inception:download_and_preprocess_flowers

# run it
bazel-bin/inception/download_and_preprocess_flowers "${FLOWERS_DATA_DIR}"
