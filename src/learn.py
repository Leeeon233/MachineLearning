from sklearn import *
import numpy as np
import pandas as pd
import plotly
import minepy

'''
# 读取训练集
train_info = pd.read_csv("../data/train3.csv", encoding="gbk")
train_set = train_info.loc[:, "男":"嗜酸细胞%"].astype(np.float32)
train_data = np.array(train_set)
# 读取目标集
target_set = train_info.loc[:,"血糖"].astype(np.float32)
target = np.array(target_set)
# 读取测试集
test_info = pd.read_csv("../data/test.csv", encoding="gbk")
test_set = test_info.loc[:, "男":"嗜酸细胞%"].astype(np.float32)
test_data = np.array(test_set)
# 表头
thead = train_set.head(0)
'''


#自测
train_info = pd.read_csv("../data/train3.csv", encoding="gb2312")
   # .drop(["*总蛋白","白蛋白","*球蛋白","白球比例","红细胞平均体积","红细胞平均血红蛋白浓度"],axis=1)

train_data = np.array(train_info.loc[:5200, "男":"嗜酸细胞%"].astype(np.float32))

test_data = np.array(train_info.loc[5201:, "男":"嗜酸细胞%"].astype(np.float32))
# 读取目标集
train_target = np.array(train_info.loc[:5200, "血糖"].astype(np.float32))
#test_info = pd.read_csv("../data/test.csv", encoding="gb2312")
    #.drop(["*总蛋白","白蛋白","*球蛋白","白球比例","红细胞平均体积","红细胞平均血红蛋白浓度"],axis=1)
#test_data = np.array(test_info.loc[:, "男":"嗜酸细胞%"].astype(np.float32))
test_target = np.array(train_info.loc[5201:, "血糖"].astype(np.float32))
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

lab_enc = preprocessing.LabelEncoder()
# 缺失值计算
train_data = preprocessing.Imputer().fit_transform(train_data)
test_data = preprocessing.Imputer().fit_transform(test_data)

#train_target = lab_enc.fit_transform(train_target)
#test_target = lab_enc.fit_transform(test_target)
#test_data = preprocessing.Imputer().fit_transform(test_data)

# plotly.plotly.iplot([trace])

#相关系数法
'''
data = feature_selection.SelectKBest(feature_selection.chi2, k=20).fit_transform(train_data, y=target)
print(data)
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
a = feature_selection.SelectFromModel(ensemble.GradientBoostingClassifier(max_features="sqrt"))
#b = lab_enc.fit_transform()
#c = lab_enc.fit_transform(target)
data= a.fit_transform(X=train_data, y=target)



result = pd.DataFrame(data=data)
result.to_csv("../data/11.csv")
'''
def meval(x, y):
    _sum = 0
    for i in range(x.__len__() - 1):
        _sum = _sum + (x[i] - y[i]) * (x[i] - y[i])
    return _sum / x.__len__() / 2


ranforest = ensemble.RandomForestRegressor()
ranforest.fit(train_data, train_target)

results = list(ranforest.predict(test_data))
result = pd.DataFrame(data=results)
print(result)
print("================_eval===========")
print(meval(results, test_target))
print("===================evaluate===========")
print(ranforest.score(test_data,test_target))
#result = pd.DataFrame(data=results)
#result.to_csv("../data/result3.csv")


