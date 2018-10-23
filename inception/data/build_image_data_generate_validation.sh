# Create the output and temporary directories.
DATA_DIR="/home/robin/Dataset/flowers"
SCRATCH_DIR="${DATA_DIR}/raw-data"
mkdir -p "${SCRATCH_DIR}"

cd "${DATA_DIR}"


# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}/train"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/validation"


# Expands the data into the flower_photos/ directory and rename it as the
# train directory.
tar xf flower_photos.tgz
rm -rf "${TRAIN_DIRECTORY}" "${VALIDATION_DIRECTORY}"
mv flower_photos "${TRAIN_DIRECTORY}"



# list of 5 labels: daisy, dandelion, roses, sunflowers, tulips
LABELS_FILE="${SCRATCH_DIR}/labels.txt"
ls -1 "${TRAIN_DIRECTORY}" | grep -v 'LICENSE' | sed 's/\///' | sort > "${LABELS_FILE}"

# Generate the validation data set.
while read LABEL; do
  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}/${LABEL}"
  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}/${LABEL}"

  # Move the first randomly selected 100 images to the validation set.
  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -10)
  for IMAGE in ${VALIDATION_IMAGES}; do
    mv -f "${TRAIN_DIRECTORY}/${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
  done
done < "${LABELS_FILE}"

# Build the TFRecords version of the image data.

# Build the TFRecords version of the image data.
OUTPUT_DIRECTORY="${SCRATCH_DIR}"
python build_image_data.py \
  --train_directory=/home/robin/Dataset/flowers/raw-data/train \
  --validation_directory=/home/robin/Dataset/flowers/raw-data/validation \
  --output_directory=/home/robin/Dataset/flowers/raw-data \
  --labels_file=/home/robin/Dataset/flowers/raw-data/labels.txt



