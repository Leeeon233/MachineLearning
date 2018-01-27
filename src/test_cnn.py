import tensorflow as tf
from tensorflow.python.ops.variables import Variable
import numpy as np
from PIL import Image

resize_width = 96
resize_height = 128
label_num = 152

def get_all_image_data(file_name):
    """
    :param img_height: 图片高度
    :param img_width:  图片宽度
    :param file_names: 图片文件名
    :return: 图片矩阵大小为  [N，img_height,img_width,1]
    """
    result = []
    print("正在读取图片,文件名:   {}".format(file_name))
    image = Image.open(file_name).convert('L')
    image = image.resize((resize_width, resize_height),Image.ANTIALIAS)
    image = np.array(image).reshape([resize_height,resize_width,1])
    #image = np.reshape(image)
    #image = tf.image.decode_jpeg(file_name, channels=1)
    #image = tf.image.resize_images(image, size=[resize_height, resize_width]).eval()
    print(np.shape(image))
    #image = tf.reshape(image,[resize_height, resize_width, 1])
    #print(np.shape(image))
    # image = np.array(tf.reshape(image, shape=[-1])).tolist()
    # print("shape:  {}".format(np.shape(image)))
    # image = tf.reshape(image, shape=img_width * img_height)
    # print("shape:  {}".format(np.shape(image)))
    result.append(image)
    return result


#x = tf.placeholder(tf.float32, shape=[None, resize_height, resize_width, 1],name="x")  # 输入的数据占位符
#y_ = tf.placeholder(tf.float32, shape=[None, label_num],name="y_")  # 输入的标签占位符
#keep_prob = tf.placeholder("float")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 根据shape初始化bias变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def model():
    conv1 = tf.layers.conv2d(x,
                             filters=32,
                             kernel_size=5,
                             strides=1,
                             activation=tf.nn.relu,
                             name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)
    conv2 = tf.layers.conv2d(pool1,
                             filters=64,
                             kernel_size=5,
                             strides=1,
                             activation=tf.nn.relu,
                             name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 38976])
    # 全连接层
    dense = tf.layers.dense(inputs=pool2_flat, units=label_num, activation=tf.nn.relu)
    # dropout 曾
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    print(dropout)

    W_fc3 = weight_variable([152, label_num])
    print(W_fc3)
    b_fc3 = bias_variable([label_num,])

    y_conv = tf.matmul(dropout, W_fc3) + b_fc3
    print(y_)
    print(y_conv)
    for k, v in locals().items():
        if type(v) is Variable or type(v) is tf.Tensor:
            print("{0}: {1}".format(k, v))

    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse

"""
声明variable和op
初始化op声明
"""
# 目录
SAVE_PATH = '../muct_data/mm/model.ckpt'
# 创建saver 对象
saver = tf.train.import_meta_graph('../muct_data/mm/model.ckpt.meta')

with tf.Session() as sess:
    # sess.run(init_op)#可以执行或不执行，restore的值会override初始值
    saver.restore(sess, SAVE_PATH)
    graph = tf.get_default_graph()

    x = graph.get_operation_by_name("x")
    print(x)
    y_conv = graph.get_operation_by_name("y_conv")
    print(y_conv)

    res = sess.run(y_conv,feed_dict={x: get_all_image_data("../muct_data/jpg/i000qa-fn.jpg")})
    print(res)