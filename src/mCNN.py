# 基于muct数据集 人脸特征检测CNN
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# 目录
img_path = "../muct_data/jpg/"
label_file = "../muct_data/mark/muct76.csv"  # 标记文件  .csv
# 原始图片宽高
img_width = 480
img_height = 640
# 目标图片宽高
resize_width = 96
resize_height = 128
label_num = 153


# 图片预处理
def getImageFiles(file_dir):  # 特定类型的文件 jpg
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(root, file))
    return L


imagefiles = getImageFiles(img_path)  # tf.read_file("../muct_data/jpg/i000qa-fn.jpg")

# 获取所有图片
imagefiles = imagefiles[:300]
file_queue = tf.train.string_input_producer(imagefiles, shuffle=False, num_epochs=1)  # 创建输入队列
image_reader = tf.WholeFileReader()
file_name, image = image_reader.read(file_queue)

all_images = tf.image.decode_jpeg(image, channels=1)
all_images = tf.image.resize_images(all_images, (resize_height, resize_width))

all_image_data = tf.placeholder(dtype=tf.float32, shape=[None, resize_width * resize_height])
all_lable_data = tf.placeholder(dtype=tf.float32, shape=[None, label_num])

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # The Coordinator helps multiple threads stop
    # together and report exceptions to a program that waits for them to stop.
    i = 0
    image_data = []
    try:
        while not coord.should_stop():
            while True:
                i += 1
                data = sess.run(all_images)
                data = tf.reshape(data, shape=[12288])
                image_data.append(data)
                print(np.shape(image_data))

    except tf.errors.OutOfRangeError:
        # When done, ask the threads to stop.
        print(i)

        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        # Wait for threads to finish.
    coord.join(threads)
    image_data = tf.reshape(image_data, shape=[300, 12288])
    print(np.shape(image_data))
