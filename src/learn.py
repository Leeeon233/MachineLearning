from sklearn import *
import numpy as np
import pandas as pd
import plotly
import minepy

# 读取训练集
train_info = pd.read_csv("../data/train.csv", encoding="gbk")
train_set = train_info.loc[:, "男":"嗜碱细胞%"].astype(np.float32)
train_data = np.array(train_set)
# 读取目标集
target_set = train_info["血糖"].astype(np.float32)
target = np.array(target_set)
# 读取测试集
test_info = pd.read_csv("../data/test.csv", encoding="gbk")
test_set = test_info.loc[:, "男":"嗜碱细胞%"].astype(np.float32)
test_data = np.array(test_set)
# 表头
thead = train_set.head(0)
'''
for i in thead:
    print("标题：" + str(i) + " 方差：" + str(np.std(train_set[i],ddof=1)))

#plotly.tools.set_credentials_file(username='leeeon', api_key='gfdwu8AQ91kTlcCLGpSN')

trace = plotly.graph_objs.Scatter(
    x=train_set["白蛋白"],
    y=target,
    mode='markers',
    title="1"
)
'''


# plotly.plotly.iplot([trace])
# 缺失值计算
train_data = preprocessing.Imputer().fit_transform(train_data)
test_data = preprocessing.Imputer().fit_transform(test_data)
#相关系数法
'''
a = feature_selection.SelectKBest(feature_selection.chi2, k=20).fit_transform(train_data, y=target)
print(a)
'''

#方差选择
'''
a= feature_selection.VarianceThreshold(threshold=1).fit_transform(train_data)
print(np.shape(a))
'''
#互信息法
'''
a = feature_selection.SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=20).fit_transform(train_data, target)
print(a)
'''
#GBDT
'''
a = feature_selection.SelectFromModel(ensemble.GradientBoostingClassifier())
#b = lab_enc.fit_transform()
#c = lab_enc.fit_transform(target)
a.fit_transform(X=train_data, y=target)
'''

ranforest = ensemble.RandomForestRegressor(max_features=6)
ranforest.fit(train_data, target)

results = list(ranforest.predict(test_data))

result = pd.DataFrame(data=results)
result.to_csv("../data/result3.csv")
