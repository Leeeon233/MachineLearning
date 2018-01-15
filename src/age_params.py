import numpy as np
import pandas as pd
import plotly.tools as pt
import plotly.plotly as py
import plotly.graph_objs as go

# 读取训练集
train_info = pd.read_csv("../data/train.csv", encoding="gbk")
train_set = train_info.loc[:, "年龄":].astype(np.float32)
train_data = np.array(train_set)
# 表头
thead = train_set.head(0)

age = train_set["年龄"]
train_set = train_set.drop(["年龄"], axis=1)




def addP(y, name):
    p = go.Bar(
        x=age,
        y=y,
        name=name
    )
    return p


layout = go.Layout(
    barmode='group'
)

data = []

for var in thead:
    if var != "年龄":
        data.append(addP(train_set[var], var))

fig = go.Figure(data=[addP(train_set["尿酸"], "尿酸")], layout=layout)
py.plot(fig, filename='年龄关系')
