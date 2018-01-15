import pandas as pd
import tensorflow as tf
import numpy as np

'''
tf_train = tf.placeholder(tf.float32,[None,39])
tf_target = tf.placeholder(tf.float32,[None,1])
tf_test = tf.placeholder(tf.float32,[None,1])
'''

# 读取训练集
train_info = pd.read_csv("../data/train.csv", encoding="gb2312")
train_data = np.array(train_info.loc[:, "男":"嗜碱细胞%"].astype(np.float32))
# 读取目标集
target = np.array(train_info["血糖"].astype(np.float32))
# 读取测试集
test_info = pd.read_csv("../data/test.csv", encoding="gb2312")
test_data = np.array(test_info.loc[:, "男":"嗜碱细胞%"].astype(np.float32))

# 构建随机森林
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_classes=1, num_features=6, regression=True,
    num_trees=100, max_nodes=1000)

classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)

classifier.fit(x=train_data, y=target, steps=100)
classifier.evaluate(x=train_data, y=target, steps=10)

# 结果
results = list(classifier.predict(x=test_data))
result = pd.DataFrame(data=results)
result.to_csv("../data/result5.csv")
