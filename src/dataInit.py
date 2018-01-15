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

# 获取特殊血糖值
# th 阈值
hh = 100
hl = 10
lh = 8
ll = 5


def xuetang(ll=5, lh=8,hl=10,hh=100):
    data_high = train_set.loc[hl <= train_set["血糖"]].loc[hh >= train_set["血糖"]]
    data_low = train_set.loc[ll <= train_set["血糖"]].loc[lh >= train_set["血糖"]]

    # 计算各项评价值
    avglist_high = list()
    avglist_low = list()

    for i in thead:
        avglist_high.append(np.mean(data_high[i]))
        avglist_low.append(np.mean(data_low[i]))
    return draw_gap(avglist_low, avglist_high,"low:"+str(ll)+"~"+str(lh)+"  high:"+str(hl)+"~"+str(hh))


# 绘制图像
# pt.set_credentials_file(username='leeeon', api_key='gfdwu8AQ91kTlcCLGpSN')
def draw_gap(avglist_low,avglist_high,name):
    gap = go.Bar(
        x=list(thead),
        y=list(map(lambda x: np.abs(x[0] - x[1]) / x[1] * 10, zip(avglist_high, avglist_low))),
        name=name
    )
    return gap


data = [xuetang(4,5,5,6),xuetang(6,7,7,8),xuetang(5,9,10,100)]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='参数平均值')
