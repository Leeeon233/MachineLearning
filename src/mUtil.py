import pandas as pd
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.python.ops.variables import Variable

# 目录
SAVE_PATH = '../muct_data/mm/model.ckpt'
img_path = "../muct_data/jpg/"
label_file = "../muct_data/mark/muct76.csv"  # 标记文件  .csv
# 原始图片宽高
img_width = 480
img_height = 640
# 目标图片宽高
resize_width = 96
resize_height = 128
label_num = 152
# 总数据
TOTAL = 3000
VALIDATION_SIZE = 2000  # 训练集大小
EPOCHS = 10  # 迭代次数
BATCH_SIZE = 100  # 每个batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 10  # 控制early stopping的参数


def getImageFiles(file_dir):  # 特定类型的文件 jpg
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(root, file))
    return L


imagefiles = getImageFiles(img_path)


def get_all_image_data(file_names):
    """
    :param img_height: 图片高度
    :param img_width:  图片宽度
    :param file_names: 图片文件名
    :return: 图片矩阵大小为  [N，img_height,img_width,1]
    """
    result = []
    print("=============将图片转化为矩阵==============")
    i = 0
    for file_name in file_names:
        i += 1
        print("正在读取第{}张图片,文件名:   {}".format(i, file_name))
        image = Image.open(file_name).convert('L')
        image = image.resize((resize_width, resize_height), Image.ANTIALIAS)
        image = np.array(image).reshape([resize_height, resize_width, 1])
        # image = np.reshape(image)
        # image = tf.image.decode_jpeg(file_name, channels=1)
        # image = tf.image.resize_images(image, size=[resize_height, resize_width]).eval()
        print(np.shape(image))
        # image = tf.reshape(image,[resize_height, resize_width, 1])
        # print(np.shape(image))
        # image = np.array(tf.reshape(image, shape=[-1])).tolist()
        # print("shape:  {}".format(np.shape(image)))
        # image = tf.reshape(image, shape=img_width * img_height)
        # print("shape:  {}".format(np.shape(image)))
        result.append(image)
    # result = np.reshape(np.array(result), newshape=[-1, resize_height, resize_width, 1])
    print(np.shape(result))
    print("Images矩阵shape:  {}".format(np.shape(result)))
    return result


def get_all_label_data(imagefiles, label_file, num):
    """
    获取标签数据\n
    :param labels: 标签pandas对象
    :param num: 标签数量
    :return: 标签矩阵
    """
    labels = pd.read_csv(label_file)
    result = []
    i = 0
    for image in imagefiles:
        i += 1
        file_name = os.path.basename(image)[:-4]
        print("读取第 {} 个标签信息，文件名 {} ".format(i, file_name))
        label = np.array(labels.loc[labels["name"] == file_name])
        label = label.reshape(-1)[1:]
        result.append(label)
    print("Labels矩阵shape:  {}".format(np.shape(result)))
    return result


# 构建网络
# 根据给定的shape定义并初始化卷积核的权值变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 根据shape初始化bias变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W):
    return tf.layers.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', data_format="channels_last", kernel_size=4)


x = tf.placeholder(tf.float32, shape=[None, resize_height, resize_width, 1], name="x")  # 输入的数据占位符
y_ = tf.placeholder(tf.float32, shape=[None, label_num], name="y_")  # 输入的标签占位符
keep_prob = tf.placeholder("float")


def model():
    conv1 = tf.layers.conv2d(x,
                             filters=32,
                             kernel_size=5,
                             strides=1,
                             activation=tf.nn.relu,
                             name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2,name="pool1")
    conv2 = tf.layers.conv2d(pool1,
                             filters=64,
                             kernel_size=5,
                             strides=1,
                             activation=tf.nn.relu,
                             name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2,name="pool2")
    pool2_flat = tf.reshape(pool2, [-1, 38976])
    # 全连接层
    dense = tf.layers.dense(inputs=pool2_flat, units=label_num, activation=tf.nn.relu, name="dense")
    # dropout 曾
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    print(dropout)

    W_fc3 = weight_variable([152, label_num])
    print(W_fc3)
    b_fc3 = bias_variable([label_num, ])

    y_conv = tf.add(tf.matmul(dropout, W_fc3), b_fc3, name="y_conv")
    '''
    for k, v in locals().items():
        if type(v) is Variable or type(v) is tf.Tensor:
            print("{0}: {1}".format(k, v))
    '''

    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse


'''
def model():
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([2, 2, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([11 * 11 * 128, 500])
    b_fc1 = bias_variable([500])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([500, 500])
    b_fc2 = bias_variable([500])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([500, 30])
    b_fc3 = bias_variable([30])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse

'''


def save_model(saver, sess, save_path):
    path = saver.save(sess, save_path)
    print('model save in :{0}'.format(path))


# 初始化变量
all_image_data = get_all_image_data(imagefiles[:TOTAL])
all_label_data = get_all_label_data(imagefiles[:TOTAL], label_file=label_file, num=label_num)

X_train = all_image_data[:VALIDATION_SIZE]
y_train = all_label_data[:VALIDATION_SIZE]
X_valid = all_image_data[VALIDATION_SIZE:]
y_valid = all_label_data[VALIDATION_SIZE:]
if __name__ == '__main__':
    with tf.Session() as sess:

        y_conv, rmse = model()
        train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

        # 变量都要初始化
        sess.run(tf.global_variables_initializer())
        best_validation_loss = 1000000.0
        current_epoch = 0
        # TRAIN_SIZE = X_train.shape[0]
        # train_index = range(TRAIN_SIZE)

        # random.shuffle(train_index)
        # X_train = X_train[train_index]
        # y_train = y_train[train_index]
        saver = tf.train.Saver()
        print(saver.saver_def.filename_tensor_name)
        # print('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
        for i in range(EPOCHS):
            # random.shuffle(train_index)  # 每个epoch都shuffle一下效果更好
            # X_train, y_train = X_train[train_index], y_train[train_index]

            for j in range((int)(np.shape(X_train)[0] / BATCH_SIZE)):
                print('epoch {0}, train {1} samples done...'.format(i, j))
                train_step.run(feed_dict={x: X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                                          y_: y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE], keep_prob: 0.5})

            # 电脑太渣，用所有训练样本计算train_loss居然死机，只好注释了。
            # train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
            validation_loss = rmse.eval(feed_dict={x: X_valid, y_: y_valid, keep_prob: 1.0})

            print('epoch {0} done! validation loss:{1}'.format(i, validation_loss * 96.0))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_epoch = i
                save_model(saver, sess, SAVE_PATH)  # 即时保存最好的结果
            elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
                print('early stopping')
                break

