#coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
import cv2
from scipy import misc
import glob
import numpy as np

cwd = os.getcwd()

def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def read_and_decode_from_tfrecord(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/class/label': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                           'image/class/text': tf.FixedLenFeature([], dtype=tf.string),
                                       })

    image = decode_jpeg(features['image/encoded'])
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    

    return image, label

#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(image, label, batch_size,crop_size = 224):
    #数据扩充变换
    print('get_batch: ',image)  
    print('get_batch: ',label)
    #distorted_image = tf.reshape(image,[224,224,3])
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪
    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转
#     distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
#     distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化
    distorted_image = tf.image.per_image_standardization(distorted_image)
    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,
                                                 num_threads=16,capacity=2000,min_after_dequeue=1000)
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)
#     images = tf.cast(images, tf.float32)
    # 调试显示
    #tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

def test():
    tf_train_file = "demo_picture/validation-00000-of-00064"
    img, label = read_and_decode_from_tfrecord(tf_train_file)#parse the tfrecode

    #初始化所有的op
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #启动队列
        coord=tf.train.Coordinator()     
        threads= tf.train.start_queue_runners(coord=coord)
        
        
        for i in range(5):
            image_np,label_np=sess.run([img,label])#每调用run一次，那么
            plt.imshow(image_np)
            plt.title('label name:'+str(label_np))
            plt.show()
#             cv2.imshow("decode",image_np)
#             cv2.waitKey(0)
            
        coord.request_stop()     
        coord.join(threads)
test()
