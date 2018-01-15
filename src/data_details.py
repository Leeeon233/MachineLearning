import numpy as np
import pandas as pd
import plotly.tools as pt
import plotly.plotly as py
import plotly.graph_objs as go

# 读取训练集
train_info = pd.read_csv("../data/train3.csv", encoding="gbk")
train_set = train_info.loc[:, "年龄":].astype(np.float32)
train_data = np.array(train_set)
# 表头
thead = train_set.head(0)

ages = train_set["年龄"]

avgs = []

for age in range(100):
    temp = train_set.loc[train_set["年龄"] == age]
    col = temp.shape[0]
    if temp.shape[0] > 0:
        li = []
        avg = 0
        for var in thead:
            if var != "年龄":
                nn = 0
                for shu in range(col):
                    t = temp[var]
                    if str(t.values[shu]) != 'nan':
                        avg += t.values[shu]
                        nn += 1

                if nn != 0:
                    avg = avg / nn

                li.append(avg)
                avg = 0
        avgs.append(li)
    else:
        avgs.append(-1)
print(avgs)

#train_info = pd.read_csv("../data/test.csv", encoding="gbk")

t = train_info.loc[:, "年龄":].astype(np.float32)

for i in range(5642):#999
    for j in range(17):
        temp = t.iloc[i][j + 1]
        if str(temp) == 'nan':
            _age = np.int(t.iloc[i][0])
            if type(avgs[_age]) != int :
                t.iloc[i][j + 1] = avgs[_age][j]

result = pd.DataFrame(data=t)
result.to_csv("../data/train_im_1.csv", encoding="gbk")
