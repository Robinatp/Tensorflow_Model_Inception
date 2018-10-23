# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import matplotlib.pyplot as plt
import cv2

import os
import sys

# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/robin/eclipse-workspace-python/TF_models/models/research/inception"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from flowers_data import *
from image_processing import *
from imagenet_data import  *

tf.app.flags.DEFINE_string('labels_file', '/home/robin/Dataset/imaget/output_tfrecord/labels.txt', 'Labels file')
tf.app.flags.DEFINE_string('data_dir', '/home/robin/Dataset/imaget/output_tfrecord',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
FLAGS = tf.app.flags.FLAGS

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == '__main__':
    
    dataset = ImagenetData(subset="validation")
#     images, labels = distorted_inputs(dataset)
    images, labels = inputs(dataset)
    labels -= 1
    print(images,labels)

    classes = load_labels(FLAGS.labels_file)
    print(classes)
    
    merge_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("logs",tf.get_default_graph())
   
   
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord=tf.train.Coordinator()     
        threads= tf.train.start_queue_runners(coord=coord)

        for i in range(10):
            image_np,label_np,summary_str=sess.run([images, labels,merge_op])
            summary_writer.add_summary(summary_str)
#             plt.imshow(image_np[0,:,:,:])
#             plt.title('label name:'+classes[label_np[0]-1])
#             plt.show()
            
            cv2.imshow('label name:',cv2.cvtColor(image_np[0,:,:,:],cv2.COLOR_RGB2BGR))
            print(classes[label_np[0]-1])
            cv2.waitKey(0)
            
            
            
        coord.request_stop()     
        coord.join(threads)