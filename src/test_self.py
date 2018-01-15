import pandas as pd
import tensorflow as tf
import numpy as np

# 读取训练集
train_info = pd.read_csv("../data/train.csv", encoding="gb2312")
train_data = np.array(train_info.loc[:4800, "男":"嗜碱细胞%"].astype(np.float32))


test_data = np.array(train_info.loc[4801:, "男":"嗜碱细胞%"].astype(np.float32))
# 读取目标集
train_target = np.array(train_info.loc[:4800, "血糖"].astype(np.float32))

test_target = np.array(train_info.loc[4801:, "血糖"].astype(np.float32))

train_info2 = pd.read_csv("../data/train2.csv", encoding="gb2312")
train_data2 = np.array(train_info.loc[:4800, "男":"嗜碱细胞%"].astype(np.float32))

# 构造input_fn
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": train_data}, train_target, batch_size=32, num_epochs=100)


# 构造评价函数
def meval(x, y):
    _sum = 0
    for i in range(x.__len__() - 1):
        _sum = _sum + (x[i] - y[i]) * (x[i] - y[i])
    return _sum / x.__len__() / 2


# 构建随机森林
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_classes=1, num_features=6, regression=True,
    num_trees=100, max_nodes=10000)

classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)

classifier.fit(input_fn=input_fn, steps=1000)
# classifier.evaluate(x=train_data, y=train_target, steps=10)


# 结果
results = list(classifier.predict(x=test_data))
result = pd.DataFrame(data=results)["scores"]

result.to_csv("../data/result6.csv", encoding="gbk")
print("================meval===========")
print(meval(result, test_target))
print("===================evaluate===========")
print(classifier.evaluate(input_fn=input_fn, steps=10))
# result = pd.DataFrame(data=results)
# result.to_csv("../data/result.csv")
